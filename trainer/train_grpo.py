import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import warnings
import torch
from torch import optim
from dataset.lm_dataset import RLAIFDataset
from trainer.cli import (new_parser, add_model_args, add_run_args, add_data_args, add_init_args, add_tracking_args, add_rl_args,
                         add_group_policy_args, config_from_args)
from trainer.engine import start_run, restore, make_sampler, run_epochs, finish_run, is_save_step
from trainer.losses import token_log_probs, group_advantages, truncate_after_eos, group_policy_loss, masked_row_mean
from trainer.rl_common import (score_responses, load_policy_and_reference, build_rollout_engine, count_batches, cosine_scheduler,
                               steps_per_run, optimizer_step, save_policy, log_samples, unwrap_ddp, wrap_policy)
from trainer.trainer_utils import Logger, is_main_process, LMForRewardModel

warnings.filterwarnings('ignore')


class GRPOTrainer:
    """Group Relative Policy Optimization: sample num_generations responses per prompt, score them, and push up the
    log-probs of responses that beat their group's mean (GRPO clipped ratio or CISPO), with a KL leash to the reference."""

    def __init__(self, args, lm_config, model, ref_model, reward_model, rollout_engine, tokenizer, optimizer, scheduler, autocast_ctx, wandb=None):
        self.args, self.lm_config, self.model, self.ref_model = args, lm_config, model, ref_model
        self.reward_model, self.rollout_engine, self.tokenizer = reward_model, rollout_engine, tokenizer
        self.optimizer, self.scheduler, self.autocast_ctx, self.wandb = optimizer, scheduler, autocast_ctx, wandb

    def response_log_probs(self, model, outputs, full_mask, logp_pos):
        return token_log_probs(model(outputs, attention_mask=full_mask).logits[:, :-1, :], outputs[:, 1:]).gather(1, logp_pos)

    def train_step(self, batch, step):
        args, tokenizer, device = self.args, self.tokenizer, self.args.device
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout = self.rollout_engine.rollout(prompt_ids=prompt_inputs["input_ids"], attention_mask=prompt_inputs["attention_mask"],
                                              num_generations=args.num_generations, max_new_tokens=args.max_gen_len, temperature=0.8)
        outputs, completion_ids = rollout.output_ids, rollout.completion_ids  # [B*G, P+R], [B*G, R]
        old_per_token_logps = rollout.per_token_logps.to(device).detach()
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = rollout.prompt_lens.to(device).unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=device).unsqueeze(0)
        rewards = score_responses(prompts, rollout.completions, self.reward_model, args.num_generations, device).to(device)  # [B*G]

        with self.autocast_ctx:
            res = unwrap_ddp(self.model)(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if self.lm_config.use_moe else torch.tensor(0.0, device=device)
            per_token_logps = token_log_probs(res.logits[:, :-1, :], outputs[:, 1:]).gather(1, logp_pos)
        with torch.no_grad():
            ref_per_token_logps = self.response_log_probs(self.ref_model, outputs, full_mask, logp_pos)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            log_samples(step, prompts, rollout.completions, rewards, args.num_generations)

        advantages, _ = group_advantages(rewards, args.num_generations)
        pad_mask = rollout.completion_mask.to(device).bool()
        completion_mask = truncate_after_eos(pad_mask, (completion_ids == tokenizer.eos_token_id) & pad_mask).int()  # [B*G, R]
        per_token_loss = group_policy_loss(per_token_logps, old_per_token_logps, ref_per_token_logps, advantages,
                                           args.loss_type, args.beta, args.epsilon, args.epsilon_high)
        policy_loss = masked_row_mean(per_token_loss, completion_mask).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()
        if step % args.accumulation_steps == 0:
            optimizer_step(self.model, self.optimizer, self.scheduler, args.grad_clip)

        return {
            "reward": rewards.mean().item(),
            "kl_ref": ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1),
            "advantages_std": advantages.std().item(),
            "advantages_mean": advantages.mean().item(),
            "policy_loss": loss.item() * args.accumulation_steps,
            "avg_response_len": completion_mask.sum(dim=1).float().mean().item(),
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }

    def log(self, epoch, step, iters, m):
        Logger(f'Epoch:[{epoch + 1}/{self.args.epochs}]({step}/{iters}), '
               f'Reward: {m["reward"]:.4f}, KL_ref: {m["kl_ref"]:.4f}, '
               f'Adv Std: {m["advantages_std"]:.4f}, Adv Mean: {m["advantages_mean"]:.4f}, '
               f'Actor Loss: {m["policy_loss"]:.4f}, Avg Response Len: {m["avg_response_len"]:.2f}, Learning Rate: {m["learning_rate"]:.8f}')
        if self.wandb and is_main_process(): self.wandb.log(m)

    def train_epoch(self, epoch, loader, iters, start_step=0):
        args, step = self.args, start_step
        for step, batch in enumerate(loader, start=start_step + 1):
            metrics = self.train_step(batch, step)
            if step % args.log_interval == 0 or step == iters:
                self.log(epoch, step, iters, metrics)
            if is_save_step(step, iters, args.save_interval):
                save_policy(args, self.lm_config, self.model, self.optimizer, epoch, step, self.wandb, scheduler=self.scheduler)
            if step % args.save_interval == 0 or step == iters:
                self.rollout_engine.update_policy(self.model)
        if step > start_step and step % args.accumulation_steps != 0:
            optimizer_step(self.model, self.optimizer, self.scheduler, args.grad_clip)


def build_parser():
    parser = new_parser("MiniMind GRPO (Group Relative Policy Optimization)")
    add_run_args(parser, save_weight='grpo', epochs=1, batch_size=2, learning_rate=3e-7, accumulation_steps=1, log_interval=1, save_interval=10)
    add_model_args(parser)
    add_data_args(parser, data_path="../dataset/rlaif.jsonl", max_seq_len=768, max_seq_len_help="Prompt最大长度")
    add_group_policy_args(parser, num_generations=6)
    add_init_args(parser, from_weight='full_sft')
    add_tracking_args(parser, wandb_project="MiniMind-GRPO")
    add_rl_args(parser, max_gen_len=1024, thinking_ratio=0.9, sglang_shared_path="./sglang_ckpt_grpo")
    return parser


def main(argv=None, reward_model=None):
    args = build_parser().parse_args(argv)
    lm_config = config_from_args(args, max_seq_len=args.max_seq_len + args.max_gen_len)
    ctx = start_run(args, lm_config, args.save_weight, f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}")
    model, ref_model, tokenizer = load_policy_and_reference(lm_config, args)
    reward_model = reward_model or LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    rollout_engine = build_rollout_engine(args, model, tokenizer, ctx.autocast_ctx)  # 可插拔替换，只负责 policy 推理
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    sampler = make_sampler(train_ds)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = cosine_scheduler(optimizer, steps_per_run(count_batches(train_ds, args, sampler), args), args.learning_rate)
    start_epoch, start_step = restore(ctx.ckp_data, model=model, optimizer=optimizer, scheduler=scheduler)
    model = wrap_policy(model, args, ctx.local_rank, rollout_engine)
    trainer = GRPOTrainer(args, lm_config, model, ref_model, reward_model, rollout_engine, tokenizer, optimizer, scheduler, ctx.autocast_ctx, ctx.wandb)
    run_epochs(args, train_ds, sampler, start_epoch, start_step, trainer.train_epoch)
    finish_run()


if __name__ == "__main__":
    main()
