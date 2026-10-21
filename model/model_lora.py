import torch
from torch import nn


# 定义Lora网络结构
class LoRA(nn.Module):
    """Low-rank adapter: y = B(A(x)). B starts at zero, so a freshly attached adapter leaves the model unchanged."""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def _strip_module_prefix(name):
    return name[7:] if name.startswith('module.') else name


def apply_lora(model, rank=16):
    """Attach a LoRA adapter to every square nn.Linear (the d×d attention projections) by wrapping its forward."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {_strip_module_prefix(k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def lora_state_dict(model):
    """Only the adapter weights (fp16, CPU), keyed '<linear name>.lora.A.weight' / '.lora.B.weight'."""
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            state_dict.update({f'{_strip_module_prefix(name)}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()})
    return state_dict


def save_lora(model, path):
    torch.save(lora_state_dict(model), path)


def merge_lora(model, lora_path, save_path):
    """Fold W += B·A into every adapted Linear and save a plain (adapter-free) fp16 state dict."""
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
