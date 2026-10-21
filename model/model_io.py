"""Loading and saving MiniMind weights: one place for wrapper unwrapping, half-precision saving and inference loading."""
import os
import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def unwrap_model(model):
    """Strip DistributedDataParallel (.module) and torch.compile (._orig_mod) wrappers."""
    model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(model, '_orig_mod', model)


def half_state_dict(model):
    """CPU fp16 copy of the unwrapped model's state dict — the format of every released *.pth file."""
    return {k: v.half().cpu() for k, v in unwrap_model(model).state_dict().items()}


def atomic_save(obj, path):
    """torch.save through a temp file so an interrupted save never leaves a truncated checkpoint."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def weight_path(directory, prefix, config: MiniMindConfig):
    """e.g. ('../out', 'full_sft', config) -> '../out/full_sft_768_moe.pth'."""
    return f'{directory}/{config.weight_name(prefix)}.pth'


def load_minimind(load_from, weight_file=None, config: MiniMindConfig = None, lora_file=None, device='cpu'):
    """Inference model + tokenizer.

    load_from containing 'model' means native torch weights: build MiniMindForCausalLM(config), load weight_file
    strictly and optionally attach a LoRA adapter. Any other path is a transformers-format folder.
    """
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if 'model' in load_from:
        model = MiniMindForCausalLM(config)
        model.load_state_dict(torch.load(weight_file, map_location=device), strict=True)
        if lora_file:
            from model.model_lora import apply_lora, load_lora
            apply_lora(model)
            load_lora(model, lora_file)
    else:
        model = AutoModelForCausalLM.from_pretrained(load_from, trust_remote_code=True)
    return model, tokenizer
