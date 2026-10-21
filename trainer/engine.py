"""Training skeleton shared by every trainer.

Run setup (`start_run`), resume (`restore`), compile/DDP wrapping (`wrap_model`), the epoch loop with mid-epoch
resume (`run_epochs`) and teardown (`finish_run`) are written once here. `SupervisedTrainer` is a Template Method:
the loop — LR schedule, gradient accumulation, clipping, logging, checkpointing — is fixed, and a subclass only says
how one batch becomes a loss (`compute_loss`) and, if needed, how to save (`save`).
"""
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_io import half_state_dict, weight_path
from trainer.trainer_utils import (Logger, get_lr, init_distributed_mode, is_main_process, lm_checkpoint, setup_seed,
                                   SkipBatchSampler)


# ───────────────────────────── run lifecycle ─────────────────────────────
@dataclass
class RunContext:
    local_rank: int
    autocast_ctx: Any
    ckp_data: Optional[dict]
    wandb: Any


def make_autocast(device, dtype):
    if "cuda" not in device: return nullcontext()
    return torch.cuda.amp.autocast(dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16)


def init_wandb(args, ckp_data, run_name):
    if not (args.use_wandb and is_main_process()): return None
    import wandb
    wandb_id = ckp_data.get('wandb_id') if ckp_data else None
    wandb.init(project=args.wandb_project, name=run_name, id=wandb_id, resume='must' if wandb_id else None)
    return wandb


def start_run(args, lm_config, resume_weight, run_name):
    """Process group, seeds, output dir, resume data, mixed precision and wandb — in that order (the order fixes RNG use)."""
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    ckp_data = lm_checkpoint(lm_config, weight=resume_weight, save_dir=args.checkpoint_dir) if args.from_resume == 1 else None
    autocast_ctx = make_autocast(args.device, args.dtype)
    return RunContext(local_rank, autocast_ctx, ckp_data, init_wandb(args, ckp_data, run_name))


def restore(ckp_data, strict=True, **components):
    """Load each named component (model=..., optimizer=..., scaler=..., critic_model=...) from resume data.
    Returns (start_epoch, start_step); (0, 0) without resume data."""
    if not ckp_data: return 0, 0
    for key, component in components.items():
        if key in ('model', 'critic_model'): component.load_state_dict(ckp_data[key], strict=strict)
        else: component.load_state_dict(ckp_data[key])
    return ckp_data['epoch'], ckp_data.get('step', 0)


def wrap_model(model, args, local_rank, compile_ok=True):
    if args.use_compile == 1 and compile_ok:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    return model


def make_sampler(dataset):
    return DistributedSampler(dataset) if dist.is_initialized() else None


def run_epochs(args, dataset, sampler, start_epoch, start_step, train_epoch, collate_fn=None):
    """Epoch loop: reshuffle with seed 42+epoch, skip the batches already done when resuming, call
    train_epoch(epoch, loader, iters, start_step)."""
    for epoch in range(start_epoch, args.epochs):
        sampler and sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(dataset)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(sampler or indices, args.batch_size, skip)
        loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step)
        else:
            train_epoch(epoch, loader, len(loader), 0)


def finish_run():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_save_step(step, iters, save_interval):
    return (step % save_interval == 0 or step == iters) and is_main_process()


def format_metrics(epoch, epochs, step, iters, metrics):
    return f'Epoch:[{epoch + 1}/{epochs}]({step}/{iters}), ' + ', '.join(
        f'{k}: {v:.8f}' if k == 'lr' else f'{k}: {v:.1f}min' if k == 'epoch_time' else f'{k}: {v:.4f}' for k, v in metrics.items())


# ───────────────────────────── supervised trainer (Template Method) ─────────────────────────────
@dataclass
class StepOutput:
    """What compute_loss returns: the full (un-accumulated) loss to backprop, the MoE aux loss, and extra metrics to log."""
    loss: torch.Tensor
    aux_loss: torch.Tensor
    metrics: Dict[str, torch.Tensor] = field(default_factory=dict)


class SupervisedTrainer:
    """Fixed training loop for losses computed from one batch. Subclasses implement compute_loss(batch)."""

    def __init__(self, args, lm_config, model, optimizer, scaler, autocast_ctx, wandb=None, clip_params=None):
        self.args, self.lm_config, self.model, self.optimizer = args, lm_config, model, optimizer
        self.scaler, self.autocast_ctx, self.wandb = scaler, autocast_ctx, wandb
        self.clip_params = clip_params  # None -> all model parameters

    # ── hooks ──
    def compute_loss(self, batch) -> StepOutput:
        raise NotImplementedError

    def save(self, epoch, step):
        """Default: fp16 weights to save_dir plus a resumable checkpoint in checkpoint_dir."""
        args = self.args
        torch.save(half_state_dict(self.model), weight_path(args.save_dir, args.save_weight, self.lm_config))
        lm_checkpoint(self.lm_config, weight=args.save_weight, model=self.model, optimizer=self.optimizer, scaler=self.scaler,
                      epoch=epoch, step=step, wandb=self.wandb, save_dir=args.checkpoint_dir)

    # ── loop ──
    def to_device(self, batch):
        if isinstance(batch, dict): return {k: v.to(self.args.device) for k, v in batch.items()}
        return [t.to(self.args.device) for t in batch]

    def set_lr(self, step, total_steps):
        lr = get_lr(step, total_steps, self.args.learning_rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.clip_params if self.clip_params is not None else self.model.parameters(), self.args.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def train_step(self, batch, step):
        """Forward + backward on one batch; steps the optimizer every accumulation_steps. Returns the loggable values."""
        accum = self.args.accumulation_steps
        out = self.compute_loss(self.to_device(batch))
        loss = out.loss / accum
        self.scaler.scale(loss).backward()
        if step % accum == 0: self.optimizer_step()
        return loss, out

    def log(self, epoch, step, iters, start_step, start_time, loss, out):
        current_loss = loss.item() * self.args.accumulation_steps
        metrics = {'loss': current_loss, **{k: float(v) for k, v in out.metrics.items()}, 'aux_loss': float(out.aux_loss)}
        lr = self.optimizer.param_groups[-1]['lr']
        eta_min = (time.time() - start_time) / max(step - start_step, 1) * (iters - step) // 60
        Logger(format_metrics(epoch, self.args.epochs, step, iters, {**metrics, 'lr': lr, 'epoch_time': eta_min}))
        if self.wandb: self.wandb.log({**metrics, "learning_rate": lr, "epoch_time": eta_min})

    def train_epoch(self, epoch, loader, iters, start_step=0):
        args, start_time, last_step = self.args, time.time(), start_step
        for step, batch in enumerate(loader, start=start_step + 1):
            last_step = step
            self.set_lr(epoch * iters + step, args.epochs * iters)
            loss, out = self.train_step(batch, step)
            if step % args.log_interval == 0 or step == iters:
                self.log(epoch, step, iters, start_step, start_time, loss, out)
            if is_save_step(step, iters, args.save_interval):
                self.model.eval()
                self.save(epoch, step)
                self.model.train()
            del loss, out
        if last_step > start_step and last_step % args.accumulation_steps != 0:  # flush a partial accumulation window
            self.optimizer_step()
