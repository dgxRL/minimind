import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from dataset.lm_dataset import RLAIFDataset
from model.model_io import weight_path
from model.model_minimind import MiniMindForCausalLM
from trainer.cli import new_parser, add_model_args, add_run_args, add_data_args, add_init_args, add_tracking_args, add_rl_args, config_from_args
from trainer.engine import start_run, restore, make_sampler, run_epochs, finish_run, is_save_step
from trainer.losses import token_log_probs, gae_advantages, masked_whiten, ppo_policy_loss, ppo_value_loss
from trainer.rl_common import (score_responses, load_policy_and_reference, build_rollout_engine, count_batches, cosine_scheduler,
                               save_policy, log_samples, unwrap_ddp)
from trainer.trainer_utils import Logger, is_main_process, LMForRewardModel

warnings.filterwarnings('ignore')


# 自定义的Critic模型，继承自MiniMindLM
class CriticModel(MiniMindForCausalLM):
    """Value model: the MiniMind backbone with a scalar value_head instead of lm_head (lm_head is kept but unused).
    Note: MiniMindModel already applies the final norm; the extra self.model.norm below is historical and kept so
    existing critic checkpoints behave the same."""
    def __init__(self, params):
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        return self.value_head(hidden_states).squeeze(-1)


class PPOTrainer:
    """Actor-critic PPO on single responses: rewards go on the last response token, GAE over the critic's values,
    then ppo_update_iters epochs of minibatch clipped updates with KL early stopping."""

    def __init__(self, args, lm_config, actor_model, critic_model, ref_model, reward_model, rollout_engine, tokenizer,
                 actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler, autocast_ctx, wandb=None):
        self.args, self.lm_config, self.tokenizer = args, lm_config, tokenizer
        self.actor_model, self.critic_model, self.ref_model, self.reward_model = actor_model, critic_model, ref_model, reward_model
        self.rollout_engine, self.autocast_ctx, self.wandb = rollout_engine, autocast_ctx, wandb
        self.actor_optimizer, self.critic_optimizer = actor_optimizer, critic_optimizer
        self.actor_scheduler, self.critic_scheduler = actor_scheduler, critic_scheduler
        self.grad_accum_step = 0

    def optimizer_step(self):
        clip_grad_norm_(self.actor_model.parameters(), self.args.grad_clip)
        clip_grad_norm_(self.critic_model.parameters(), self.args.grad_clip)
        for opt in (self.actor_optimizer, self.critic_optimizer): opt.step()
        for sched in (self.actor_scheduler, self.critic_scheduler): sched.step()
        for opt in (self.actor_optimizer, self.critic_optimizer): opt.zero_grad()

    def rollout(self, prompts, step):
        """Sample responses and build everything the update needs (all without gradients)."""
        args, tokenizer, device = self.args, self.tokenizer, self.args.device
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len, padding_side="left").to(device)
        result = self.rollout_engine.rollout(prompt_ids=enc.input_ids, attention_mask=enc.attention_mask, num_generations=1,
                                             max_new_tokens=args.max_gen_len, temperature=0.8)
        gen_out, resp_labels = result.output_ids, result.completion_ids
        prompt_lens = result.prompt_lens.to(device)
        old_resp_logp = result.per_token_logps.to(device)
        rewards = score_responses(prompts, result.completions, self.reward_model, 1, device)  # [B]

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            log_samples(step, prompts, result.completions, rewards, 1)

        B = len(prompts)
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx
        resp_pad_mask = result.completion_mask.to(device).bool()
        # a response ends at its first EOS (inclusive), else at its last non-pad token
        resp_lengths = resp_pad_mask.sum(dim=1); valid_resp = resp_lengths > 0
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        resp_lengths = torch.where(eos_mask.any(dim=1), torch.argmax(eos_mask.int(), dim=1) + 1, resp_lengths).long().clamp(min=1)
        resp_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()

        with torch.no_grad():  # Rollout阶段只需推理获取old_logp和old_values，切断梯度省显存
            old_values = unwrap_ddp(self.critic_model)(input_ids=gen_out, attention_mask=full_mask).gather(1, logp_pos) * resp_mask
            ref_logp = token_log_probs(self.ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], labels).gather(1, logp_pos)
            token_rewards = torch.zeros_like(old_resp_logp)
            token_rewards[torch.arange(B, device=device)[valid_resp], (resp_lengths - 1)[valid_resp]] += rewards[valid_resp]  # 末尾加外部奖励
            advantages, returns = gae_advantages(token_rewards, old_values, args.gamma, args.lam)
            advantages = masked_whiten(advantages, resp_mask)
        return dict(gen_out=gen_out, full_mask=full_mask, labels=labels, logp_pos=logp_pos, resp_mask=resp_mask, old_logp=old_resp_logp,
                    ref_logp=ref_logp, old_values=old_values, advantages=advantages, returns=returns, rewards=rewards, resp_lengths=resp_lengths)

    def update(self, r, step):
        """Minibatch PPO epochs over one rollout. Returns averaged diagnostics."""
        args, device = self.args, self.args.device
        B = r['gen_out'].size(0)
        mb_size = max(1, min(args.mini_batch_size, B))
        actor, critic = unwrap_ddp(self.actor_model), unwrap_ddp(self.critic_model)
        sums = dict(policy_loss=0.0, value_loss=0.0, approx_kl=0.0, kl_ref=0.0, clipfrac=0.0, aux_loss=0.0)
        count, stop_ppo = 0, False
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo: break
            b_inds = torch.randperm(B, device=device)
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]
                mask = r['resp_mask'][inds]
                values = critic(input_ids=r['gen_out'][inds], attention_mask=r['full_mask'][inds]).gather(1, r['logp_pos'][inds])
                with self.autocast_ctx:
                    res = actor(input_ids=r['gen_out'][inds], attention_mask=r['full_mask'][inds])
                    aux_loss = res.aux_loss if self.lm_config.use_moe else torch.tensor(0.0, device=device)
                    # 在 autocast 内计算 log_softmax，避免直接对 fp16/bf16 logits 计算造成额外数值偏差。
                    logp = token_log_probs(res.logits[:, :-1], r['labels'][inds]).gather(1, r['logp_pos'][inds])

                log_ratio = logp - r['old_logp'][inds]
                if args.debug_log_ratio and ppo_epoch == 0 and i == 0 and is_main_process():
                    self.log_ratio_debug(step, log_ratio.detach(), mask.bool(), actor)
                approx_kl = (0.5 * (log_ratio ** 2) * mask).sum() / mask.sum().clamp(min=1)
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():  # 同步各卡的 approx_kl，防止某卡 break 而其它卡继续导致 DDP 死锁
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True

                policy_loss, clipfrac, kl_ref = ppo_policy_loss(logp, r['old_logp'][inds], r['ref_logp'][inds], r['advantages'][inds], mask,
                                                                args.clip_epsilon, args.kl_coef)
                value_loss = ppo_value_loss(values, r['old_values'][inds], r['returns'][inds], mask, args.cliprange_value)
                # 早停时必须保证 forward-backward 闭环，故只截断 loss 不中断 DDP 通信
                total = policy_loss + args.vf_coef * value_loss + aux_loss
                (total * 0.0 if stop_ppo else total / args.accumulation_steps).backward()

                for k, v in dict(policy_loss=policy_loss, value_loss=value_loss, approx_kl=approx_kl_val, kl_ref=kl_ref.detach(),
                                 clipfrac=clipfrac, aux_loss=aux_loss).items():
                    sums[k] += v.item()
                count += 1
                self.grad_accum_step += 1
                if self.grad_accum_step % args.accumulation_steps == 0:
                    self.optimizer_step()
        if self.grad_accum_step % args.accumulation_steps != 0:
            self.optimizer_step()
        return {k: v / max(count, 1) for k, v in sums.items()}

    def log_ratio_debug(self, step, lr, mask, actor):
        """--debug_log_ratio: size of the first minibatch's log-ratio (should be ≈0 on the first PPO epoch)."""
        if not mask.any(): return
        v = lr[mask]
        Logger(f"[DBG log_ratio] step={step} max|lr|={v.abs().max().item():.6e} mean|lr|={v.abs().mean().item():.6e} "
               f"ratio_max={torch.exp(v).max().item():.6f} ratio_min={torch.exp(v).min().item():.6f} "
               f"dropout={getattr(self.lm_config, 'dropout', None)} training={actor.training}")

    def log(self, epoch, step, iters, r, stats):
        m = {"reward": r['rewards'].mean().item(), "kl_ref": stats['kl_ref'], "approx_kl": stats['approx_kl'], "clipfrac": stats['clipfrac'],
             "critic_loss": stats['value_loss'], "avg_response_len": r['resp_lengths'].float().mean().item(),
             "actor_lr": self.actor_optimizer.param_groups[0]['lr'], "critic_lr": self.critic_optimizer.param_groups[0]['lr']}
        if self.wandb is not None: self.wandb.log(m)
        Logger(f"Epoch:[{epoch + 1}/{self.args.epochs}]({step}/{iters}), "
               f"Reward: {m['reward']:.4f}, KL_ref: {m['kl_ref']:.4f}, Approx KL: {m['approx_kl']:.4f}, "
               f"ClipFrac: {m['clipfrac']:.4f}, Critic Loss: {m['critic_loss']:.4f}, "
               f"Avg Response Len: {m['avg_response_len']:.2f}, Actor LR: {m['actor_lr']:.8f}, Critic LR: {m['critic_lr']:.8f}")

    def train_epoch(self, epoch, loader, iters, start_step=0):
        args = self.args
        self.actor_model.train()
        self.critic_model.train()
        self.grad_accum_step = 0
        for step, batch in enumerate(loader, start=start_step + 1):
            r = self.rollout(batch["prompt"], step)
            stats = self.update(r, step)
            if step % args.save_interval == 0 or step == iters: self.rollout_engine.update_policy(self.actor_model)
            if is_main_process(): self.log(epoch, step, iters, r, stats)
            if is_save_step(step, iters, args.save_interval):
                # 使用 lm_checkpoint 保存完整状态（包括 critic）
                save_policy(args, self.lm_config, self.actor_model, self.actor_optimizer, epoch, step, self.wandb,
                            scheduler=self.actor_scheduler, critic_model=self.critic_model,
                            critic_optimizer=self.critic_optimizer, critic_scheduler=self.critic_scheduler)
            del r, stats


def build_parser():
    parser = new_parser("MiniMind PPO (Proximal Policy Optimization)")
    add_run_args(parser, save_weight='ppo_actor', epochs=1, batch_size=2, learning_rate=3e-7, accumulation_steps=1, log_interval=1, save_interval=10,
                 lr_help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    add_model_args(parser)
    add_data_args(parser, data_path="../dataset/rlaif.jsonl", max_seq_len=768, max_seq_len_help="Prompt最大长度")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    add_init_args(parser, from_weight='full_sft')
    add_tracking_args(parser, wandb_project="MiniMind-PPO")
    add_rl_args(parser, max_gen_len=1024, thinking_ratio=0.9, sglang_shared_path="./sglang_ckpt_ppo")
    parser.add_argument("--debug_log_ratio", action="store_true", help="打印首轮首个minibatch的log_ratio差异量级，用于核查ratio≈1是否成立")
    return parser


def main(argv=None, reward_model=None):
    args = build_parser().parse_args(argv)
    lm_config = config_from_args(args)
    ctx = start_run(args, lm_config, args.save_weight, f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}")
    actor_model, ref_model, tokenizer = load_policy_and_reference(lm_config, args)
    critic_model = CriticModel(lm_config)  # critic starts from the same weights as the actor
    critic_model.load_state_dict(torch.load(weight_path(args.from_dir, args.from_weight, lm_config), map_location=args.device), strict=False)
    critic_model = critic_model.to(args.device)
    reward_model = reward_model or LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    rollout_engine = build_rollout_engine(args, actor_model, tokenizer, ctx.autocast_ctx)
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    sampler = make_sampler(train_ds)
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    iters = count_batches(train_ds, args, sampler)
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    actor_scheduler = cosine_scheduler(actor_optimizer, total_optimizer_steps, args.learning_rate)
    critic_scheduler = cosine_scheduler(critic_optimizer, total_optimizer_steps, args.critic_learning_rate)
    start_epoch, start_step = restore(ctx.ckp_data, model=actor_model, critic_model=critic_model, optimizer=actor_optimizer,
                                      critic_optimizer=critic_optimizer, scheduler=actor_scheduler, critic_scheduler=critic_scheduler)
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)
    if dist.is_initialized():
        actor_model = DistributedDataParallel(actor_model, device_ids=[ctx.local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[ctx.local_rank])
    rollout_engine.update_policy(actor_model)
    trainer = PPOTrainer(args, lm_config, actor_model, critic_model, ref_model, reward_model, rollout_engine, tokenizer,
                         actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler, ctx.autocast_ctx, ctx.wandb)
    run_epochs(args, train_ds, sampler, start_epoch, start_step, trainer.train_epoch)
    finish_run()


if __name__ == "__main__":
    main()
