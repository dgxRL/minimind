"""Next-token sampling steps used by MiniMindForCausalLM.generate."""
import torch
from model.model_minimind import apply_repetition_penalty, apply_top_k, apply_top_p, sample_next_token


def test_top_k_keeps_k_largest():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = apply_top_k(logits.clone(), 2)
    assert torch.isfinite(out).tolist() == [[False, True, False, True, False]]


def test_top_p_keeps_smallest_nucleus_and_never_empties():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    out = apply_top_p(probs.log(), 0.7)
    assert torch.isfinite(out).tolist() == [[True, True, False, False]]  # 0.5 < 0.7, 0.5+0.3 crosses it
    out = apply_top_p(probs.log(), 0.1)
    assert torch.isfinite(out).sum() == 1                                # the top token always survives


def test_repetition_penalty():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_repetition_penalty(logits.clone(), torch.tensor([[0, 1]]), 2.0)
    torch.testing.assert_close(out, torch.tensor([[1.0, -4.0, 1.0]]))


def test_greedy_is_argmax_and_sampling_respects_filters():
    logits = torch.tensor([[0.1, 3.0, 0.2, 2.9]])
    assert sample_next_token(logits.clone(), None, do_sample=False, top_k=0, top_p=1.0).item() == 1
    torch.manual_seed(0)
    draws = {sample_next_token(logits.clone(), None, do_sample=True, top_k=2, top_p=1.0).item() for _ in range(50)}
    assert draws <= {1, 3}
