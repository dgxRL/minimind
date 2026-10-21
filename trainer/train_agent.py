import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import warnings
import torch
from torch import optim
from dataset.lm_dataset import AgentRLDataset
from trainer.agent_env import rollout_batch, calculate_rewards
from trainer.cli import (new_parser, add_model_args, add_run_args, add_data_args, add_init_args, add_tracking_args, add_rl_args,
                         add_group_policy_args, config_from_args)
from trainer.engine import start_run, restore, make_sampler, run_epochs, finish_run, is_save_step
from trainer.losses import token_log_probs, group_advantages, truncate_after_eos, group_policy_loss
from trainer.rl_common import (load_policy_and_reference, build_rollout_engine, count_batches, cosine_scheduler, steps_per_run,
                               optimizer_step, save_policy, unwrap_ddp, wrap_policy)
from trainer.rollout_engine import compute_per_token_logps
from trainer.trainer_utils import Logger, is_main_process, LMForRewardModel

warnings.filterwarnings('ignore')


def collate_fn(batch):
    return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}


def pack_trajectories(trajectories, pad_token_id, max_total_len, device):
    """Right-pad prompt+response token lists into tensors. Sequences longer than max_total_len keep their tail.
    Returns input_ids, response mask (1 = model-generated token), old per-token log-probs [N, L-1], prompt and sequence lengths."""
    packed = []
    for t in trajectories:
        ids = t.prompt_ids + t.response_ids
        mask = [0] * len(t.prompt_ids) + t.response_mask
        old_logps = [0.0] * max(len(t.prompt_ids) - 1, 0) + t.old_logps
        if len(ids) > max_total_len:
            ids, mask = ids[-max_total_len:], mask[-max_total_len:]
            old_logps = old_logps[-(len(ids) - 1):]
        prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
        packed.append((ids, mask, prompt_len, old_logps))
    seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed], device=device)
    max_len = seq_lens.max().item()
    input_ids = torch.tensor([ids + [pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed], device=device)
    prompt_lens = torch.tensor([p for _, _, p, _ in packed], device=device)
    response_mask = torch.tensor([m + [0] * (max_len - len(m)) for _, m, _, _ in packed], device=device, dtype=torch.float32)
    old_logps = torch.tensor([lp + [0.0] * ((max_len - 1) - len(lp)) for _, _, _, lp in packed], device=device, dtype=torch.float32)
    return input_ids, response_mask, old_logps, prompt_lens, seq_lens


class AgentTrainer:
    """Multi-turn tool-use RL with the GRPO/CISPO objective. Only model-generated tokens are trained; tool
    observations inserted between turns are masked out."""

    def __init__(self, args, lm_config, model, ref_model, reward_model, rollout_engine, tokenizer, optimizer, scheduler, autocast_ctx, wandb=None):
        self.args, self.lm_config, self.model, self.ref_model = args, lm_config, model, ref_model
        self.reward_model, self.rollout_engine, self.tokenizer = reward_model, rollout_engine, tokenizer
        self.optimizer, self.scheduler, self.autocast_ctx, self.wandb = optimizer, scheduler, autocast_ctx, wandb

    def train_step(self, batch, step):
        args, tokenizer, device = self.args, self.tokenizer, self.args.device
        messages_batch, tools_batch, gt_batch = batch['messages'], batch['tools'], batch['gt']
        with torch.no_grad():
            trajectories = rollout_batch(self.rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3,
                                         max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=device)
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
        input_ids, response_mask, old_per_token_logps, prompt_lens, seq_lens = pack_trajectories(trajectories, tokenizer.pad_token_id, args.max_total_len, device)
        full_mask = (input_ids != tokenizer.pad_token_id).long()
        rewards = calculate_rewards(prompts, [t.completion for t in trajectories], gt_batch, tools_batch, args.num_generations, self.reward_model,
                                    device=device, turn_outputs_batch=[t.turn_outputs for t in trajectories],
                                    unfinished_batch=[t.unfinished for t in trajectories])

        with self.autocast_ctx:
            res = unwrap_ddp(self.model)(input_ids, attention_mask=full_mask)
            aux_loss = res.aux_loss if self.lm_config.use_moe else torch.tensor(0.0, device=device)
            per_token_logps = token_log_probs(res.logits[:, :-1, :], input_ids[:, 1:])
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(self.ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

        completion_mask = response_mask[:, 1:]
        completion_mask = truncate_after_eos(completion_mask, (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool())
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            self.log_samples(step, gt_batch, trajectories, input_ids, prompt_lens, seq_lens, rewards)

        advantages, grouped_rewards = group_advantages(rewards, args.num_generations)
        per_token_loss = group_policy_loss(per_token_logps, old_per_token_logps, ref_per_token_logps, advantages,
                                           args.loss_type, args.beta, args.epsilon, args.epsilon_high)
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()
        if step % args.accumulation_steps == 0:
            optimizer_step(self.model, self.optimizer, self.scheduler, args.grad_clip)

        return {"reward": rewards.mean().item(),
                "kl_ref": ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1),
                "group_reward_std": grouped_rewards.std(dim=1, unbiased=False).mean().item(),
                "advantages_std": advantages.std().item(), "policy_loss": loss.item() * args.accumulation_steps,
                "avg_response_len": token_counts.float().mean().item(), "advantages_mean": advantages.mean().item(),
                "learning_rate": self.optimizer.param_groups[0]['lr']}

    def log_samples(self, step, gt_batch, trajectories, input_ids, prompt_lens, seq_lens, rewards):
        for i in range(len(gt_batch)):
            Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
            Logger('-' * 100)
            for j in range(self.args.num_generations):
                idx = i * self.args.num_generations + j
                plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                Logger(trajectories[idx].context)
                Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                Logger(self.tokenizer.decode(input_ids[idx, plen:slen].tolist(), skip_special_tokens=False))
                Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                Logger('=' * 100)

    def log(self, epoch, step, iters, m):
        Logger(f'Epoch:[{epoch+1}/{self.args.epochs}]({step}/{iters}), Reward:{m["reward"]:.4f}, KL:{m["kl_ref"]:.4f}, '
               f'GrpStd:{m["group_reward_std"]:.4f}, AdvStd:{m["advantages_std"]:.4f}, Loss:{m["policy_loss"]:.4f}, '
               f'AvgLen:{m["avg_response_len"]:.2f}, AdvMean:{m["advantages_mean"]:.4f}, LR:{m["learning_rate"]:.8f}')
        if self.wandb and is_main_process(): self.wandb.log(m)

    def train_epoch(self, epoch, loader, iters, start_step=0):
        args, last_step = self.args, start_step
        for step, batch in enumerate(loader, start=start_step + 1):
            last_step = step
            metrics = self.train_step(batch, step)
            if step % args.log_interval == 0 or step == iters:
                self.log(epoch, step, iters, metrics)
            if is_save_step(step, iters, args.save_interval):
                save_policy(args, self.lm_config, self.model, self.optimizer, epoch, step, self.wandb, scheduler=self.scheduler)
            if step % args.save_interval == 0 or step == iters:
                self.rollout_engine.update_policy(self.model)
        if last_step > start_step and last_step % args.accumulation_steps != 0:
            optimizer_step(self.model, self.optimizer, self.scheduler, args.grad_clip)


def build_parser():
    parser = new_parser("MiniMind Agent RL")
    add_run_args(parser, save_weight='agent', epochs=1, batch_size=2, learning_rate=3e-7, accumulation_steps=1, log_interval=1, save_interval=10)
    add_model_args(parser)
    add_data_args(parser, data_path="../dataset/agent_rl.jsonl", max_seq_len=1024, max_seq_len_help="最大序列长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    add_group_policy_args(parser, num_generations=4)
    add_init_args(parser, from_weight='full_sft')
    add_tracking_args(parser, wandb_project="MiniMind-Agent-RL")
    add_rl_args(parser, max_gen_len=768, thinking_ratio=0.1, sglang_shared_path="./sglang_ckpt_agent")
    return parser


def main(argv=None, reward_model=None):
    args = build_parser().parse_args(argv)
    lm_config = config_from_args(args, max_seq_len=args.max_seq_len + args.max_gen_len)
    ctx = start_run(args, lm_config, args.save_weight, f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}")
    model, ref_model, tokenizer = load_policy_and_reference(lm_config, args)
    reward_model = reward_model or LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    rollout_engine = build_rollout_engine(args, model, tokenizer, ctx.autocast_ctx)
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    sampler = make_sampler(train_ds)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = cosine_scheduler(optimizer, steps_per_run(count_batches(train_ds, args, sampler, collate_fn), args), args.learning_rate)
    start_epoch, start_step = restore(ctx.ckp_data, model=model, optimizer=optimizer, scheduler=scheduler)
    model = wrap_policy(model, args, ctx.local_rank, rollout_engine)
    trainer = AgentTrainer(args, lm_config, model, ref_model, reward_model, rollout_engine, tokenizer, optimizer, scheduler, ctx.autocast_ctx, ctx.wandb)
    run_epochs(args, train_ds, sampler, start_epoch, start_step, trainer.train_epoch, collate_fn=collate_fn)
    finish_run()


if __name__ == "__main__":
    main()
