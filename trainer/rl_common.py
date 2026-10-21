"""Pieces shared by the RL trainers (GRPO, PPO, agent): reward shaping, model/engine setup, optimiser step, saving."""
import math
import re

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from model.model_io import half_state_dict, weight_path
from trainer.rollout_engine import create_rollout_engine
from trainer.trainer_utils import Logger, init_model, lm_checkpoint

CHAT_TURN = re.compile(r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>", re.DOTALL)


# ───────────────────────────── rewards ─────────────────────────────
def rep_penalty(text, n=3, cap=0.5):
    """Share of repeated word n-grams, scaled to [0, cap] (cap reached when a quarter of the n-grams repeat)."""
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def parse_chat_prompt(prompt):
    """Rendered chat-template prompt -> [{'role', 'content'}] (system / user / assistant turns)."""
    return [{"role": role, "content": content.strip()} for role, content in CHAT_TURN.findall(prompt)]


def format_reward_terms(response):
    """Rule-based shaping for a plain chat response. Returns (terms to add, answer text with the think block removed):
    length in [20, 800] chars, think block length in [20, 300], exactly one </think>, and a repetition penalty."""
    terms = [0.5 if 20 <= len(response.strip()) <= 800 else -0.5]
    answer = response
    if '</think>' in response:
        thinking, answer_content = response.split('</think>', 1)
        terms.append(1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5)
        terms.append(0.25 if response.count('</think>') == 1 else -0.25)
        answer = answer_content.strip()
    terms.append(-rep_penalty(answer))
    return terms, answer


def score_responses(prompts, responses, reward_model, num_generations=1, device='cpu'):
    """Format terms + reward-model score per response; response i belongs to prompts[i // num_generations]."""
    rewards = torch.zeros(len(responses), device=device)
    with torch.no_grad():
        scores = []
        for idx, response in enumerate(responses):
            terms, answer = format_reward_terms(response)
            for term in terms: rewards[idx] += term
            scores.append(reward_model.get_score(parse_chat_prompt(prompts[idx // num_generations]), answer))
        rewards += torch.tensor(scores, device=device)
    return rewards


# ───────────────────────────── setup ─────────────────────────────
def load_policy_and_reference(lm_config, args):
    """Policy (trainable) and a frozen copy as the KL reference, both from --from_weight."""
    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.from_dir, device=args.device)
    ref_model, _ = init_model(lm_config, args.from_weight, save_dir=args.from_dir, device=args.device)
    return model, ref_model.eval().requires_grad_(False), tokenizer


def build_rollout_engine(args, policy_model, tokenizer, autocast_ctx):
    return create_rollout_engine(engine_type=args.rollout_engine, policy_model=policy_model, tokenizer=tokenizer, device=args.device,
                                 autocast_ctx=autocast_ctx, sglang_base_url=args.sglang_base_url,
                                 sglang_model_path=args.sglang_model_path, sglang_shared_path=args.sglang_shared_path)


def count_batches(dataset, args, sampler, collate_fn=None):
    return len(DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn))


def cosine_scheduler(optimizer, total_steps, learning_rate):
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate / 10)


def steps_per_run(iters, args):
    return math.ceil(iters / args.accumulation_steps) * args.epochs


def unwrap_ddp(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap_policy(model, args, local_rank, rollout_engine):
    """torch.compile / DDP the policy and point the rollout engine at the wrapped model after each wrap."""
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)
    return model


# ───────────────────────────── loop helpers ─────────────────────────────
def optimizer_step(model, optimizer, scheduler, grad_clip):
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def save_policy(args, lm_config, model, optimizer, epoch, step, wandb, **extra):
    """fp16 policy weights to save_dir + resumable checkpoint (extra: scheduler, critic_model, ...)."""
    model.eval()
    torch.save(half_state_dict(model), weight_path(args.save_dir, args.save_weight, lm_config))
    lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, epoch=epoch, step=step, wandb=wandb,
                  save_dir=args.checkpoint_dir, **extra)
    model.train()


def log_samples(step, prompts, completions, rewards, num_generations):
    """--debug_mode dump of every prompt with its sampled responses and their rewards."""
    for i in range(len(prompts)):
        Logger(f"[DEBUG] step={step}, sample[{i}]")
        Logger('-' * 100)
        Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
        Logger(prompts[i])
        Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
        for j in range(num_generations):
            idx = i * num_generations + j
            Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
            Logger(completions[idx])
            Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
            Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
        Logger('=' * 100)
