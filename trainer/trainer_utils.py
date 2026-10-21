"""
训练工具函数集合 — process/distributed helpers, LR schedule, checkpoints, model init, resumable sampler, reward model wrapper.
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel
from model.model_minimind import MiniMindForCausalLM
from model.model_io import unwrap_model, half_state_dict, atomic_save, weight_path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TOKENIZER_DIR = os.path.join(ROOT_DIR, 'model')


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """Cosine decay from 1.0·lr to 0.1·lr over total_steps."""
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """Join the torchrun process group if launched by torchrun; returns the local rank (0 when not distributed)."""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _wandb_run_id(wandb):
    if not wandb: return None
    if hasattr(wandb, 'get_run'):
        run = wandb.get_run()
        return getattr(run, 'id', None) if run else None
    return getattr(wandb, 'id', None)


def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """Save mode (model given): write <weight>.pth and <weight>_resume.pth (model, optimizer, progress, plus any extra
    kwargs such as scaler/scheduler/critic_model). Load mode (model=None): return the resume dict or None."""
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = weight_path(save_dir, weight, lm_config)
    resume_path = ckp_path[:-len('.pth')] + '_resume.pth'

    if model is not None:
        state_dict = half_state_dict(model)
        atomic_save(state_dict, ckp_path)
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': world_size(),
            'wandb_id': _wandb_run_id(wandb)
        }
        for key, value in kwargs.items():
            if value is not None:
                resume_data[key] = unwrap_model(value).state_dict() if hasattr(value, 'state_dict') else value
        atomic_save(resume_data, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = world_size()
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path=TOKENIZER_DIR, save_dir='../out', device='cuda'):
    """Build MiniMindForCausalLM(lm_config); unless from_weight == 'none', load <save_dir>/<from_weight>_<size><suffix>.pth."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        weights = torch.load(weight_path(save_dir, from_weight, lm_config), map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """Batches indices from `sampler`, dropping the first `skip_batches` batches (used to resume mid-epoch)."""
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """InternLM2 reward model wrapper: score(conversation, response) clipped to [-3, 3]."""
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
