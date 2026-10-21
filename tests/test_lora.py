import torch
from torch import nn
from conftest import tiny_model
from model.model_lora import apply_lora, load_lora, lora_state_dict, merge_lora, save_lora


def _ids():
    return torch.randint(3, 128, (2, 10), generator=torch.Generator().manual_seed(0))


def _perturb(model):
    for n, p in model.named_parameters():
        if '.lora.B' in n: p.data.normal_(0, 0.1)


def test_lora_attaches_to_square_linears_and_starts_as_identity():
    model = tiny_model().eval()
    with torch.no_grad(): before = model(_ids()).logits
    apply_lora(model, rank=4)
    adapted = [n for n, m in model.named_modules() if hasattr(m, 'lora')]
    assert adapted and all(isinstance(m, nn.Linear) and m.in_features == m.out_features for n, m in model.named_modules() if hasattr(m, 'lora'))
    with torch.no_grad(): torch.testing.assert_close(model(_ids()).logits, before)  # B = 0 at init


def test_save_load_round_trip(tmp_path):
    model = tiny_model().eval(); apply_lora(model, rank=4); _perturb(model)
    save_lora(model, tmp_path / 'lora.pth')
    assert all('.lora.' in k for k in lora_state_dict(model))
    other = tiny_model().eval(); apply_lora(other, rank=4); load_lora(other, tmp_path / 'lora.pth')
    with torch.no_grad(): torch.testing.assert_close(other(_ids()).logits, model(_ids()).logits, rtol=1e-3, atol=1e-3)  # fp16 adapters


def test_merge_matches_adapted_model(tmp_path):
    model = tiny_model().eval(); apply_lora(model, rank=4); _perturb(model)
    save_lora(model, tmp_path / 'lora.pth')
    base = tiny_model().eval(); apply_lora(base, rank=4)
    merge_lora(base, tmp_path / 'lora.pth', tmp_path / 'merged.pth')
    merged = tiny_model().eval(); merged.load_state_dict(torch.load(tmp_path / 'merged.pth'), strict=False)
    with torch.no_grad(): torch.testing.assert_close(merged(_ids()).logits, model(_ids()).logits, rtol=2e-2, atol=2e-2)  # fp16 weights
