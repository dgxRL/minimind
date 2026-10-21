"""Training objectives: values on hand-checked inputs and agreement with straightforward reference implementations."""
import math
import pytest
import torch
import torch.nn.functional as F
from trainer.losses import (distillation_loss, dpo_loss, gae_advantages, group_advantages, group_policy_loss, kl_k3, masked_mean,
                            masked_row_mean, masked_whiten, ppo_policy_loss, ppo_value_loss, token_log_probs, truncate_after_eos)


def test_token_log_probs():
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[0, 4, 2], [1, 1, 3]])
    expected = torch.stack([torch.stack([F.log_softmax(logits[b, t], -1)[labels[b, t]] for t in range(3)]) for b in range(2)])
    torch.testing.assert_close(token_log_probs(logits, labels), expected)


def test_masked_means():
    v, m = torch.tensor([[1.0, 2.0, 9.0], [4.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert masked_mean(v, m).item() == pytest.approx(7 / 3)
    torch.testing.assert_close(masked_row_mean(v, m), torch.tensor([1.5, 4.0]))
    assert masked_mean(v, torch.zeros_like(m)).item() == 0


# ── DPO / distillation ──
def test_dpo_loss_is_log2_when_policy_equals_reference():
    lp = torch.randn(4, 6)
    assert dpo_loss(lp, lp, torch.ones(4, 6), beta=0.1).item() == pytest.approx(math.log(2))


def test_dpo_loss_rewards_preferring_the_chosen_response():
    ref = torch.zeros(2, 3)
    policy = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])  # row 0 chosen, row 1 rejected
    loss = dpo_loss(ref, policy, torch.ones(2, 3), beta=0.5)
    assert loss.item() == pytest.approx(-math.log(torch.sigmoid(torch.tensor(0.5 * 6.0)).item()), rel=1e-5)
    assert dpo_loss(ref, policy.flip(0), torch.ones(2, 3), beta=0.5) > loss


def test_dpo_loss_ignores_masked_tokens():
    ref, policy = torch.zeros(2, 3), torch.tensor([[0.0, 0.0, 50.0], [0.0, 0.0, -50.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    assert dpo_loss(ref, policy, mask, beta=1.0).item() == pytest.approx(math.log(2))


def test_distillation_loss():
    s = torch.randn(5, 7)
    assert distillation_loss(s, s.clone(), temperature=2.0).item() == pytest.approx(0, abs=1e-6)
    t = torch.randn(5, 7)
    kl = F.kl_div(F.log_softmax(s / 2, -1), F.softmax(t / 2, -1), reduction='batchmean')
    assert distillation_loss(s, t, temperature=2.0).item() == pytest.approx(4 * kl.item(), rel=1e-5)


# ── group policy (GRPO / CISPO) ──
def test_group_advantages_normalise_within_each_group():
    rewards = torch.tensor([1.0, 3.0, 5.0, 5.0])
    adv, grouped = group_advantages(rewards, 2)
    assert grouped.shape == (2, 2)
    torch.testing.assert_close(adv, torch.tensor([-1.0, 1.0, 0.0, 0.0]) / (1 + 1e-4) * torch.tensor([1, 1, 0, 0]), atol=1e-4, rtol=0)


def test_truncate_after_eos():
    mask = torch.ones(3, 5, dtype=torch.bool)
    is_eos = torch.tensor([[0, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    assert truncate_after_eos(mask, is_eos).int().tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]
    fmask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    out = truncate_after_eos(fmask, torch.tensor([[0, 1, 0, 0]], dtype=torch.bool))
    assert out.dtype == torch.float32 and out.tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_kl_k3_nonnegative_and_zero_at_equality():
    a, b = torch.randn(100), torch.randn(100)
    assert (kl_k3(a, b) >= 0).all() and torch.count_nonzero(kl_k3(a, a)) == 0


def test_grpo_clipping_stops_gradient_outside_trust_region():
    old = torch.zeros(1, 2)
    logps = torch.tensor([[0.5, -0.05]], requires_grad=True)          # ratios e^0.5 (outside 1.2) and e^-0.05 (inside)
    loss = group_policy_loss(logps, old, logps.detach(), torch.tensor([1.0]), loss_type='grpo', beta=0.0, epsilon=0.2)
    loss.sum().backward()
    assert logps.grad[0, 0] == 0 and logps.grad[0, 1] != 0


def test_cispo_keeps_gradient_on_clipped_tokens():
    old = torch.zeros(1, 2)
    logps = torch.tensor([[3.0, 0.0]], requires_grad=True)            # ratio e^3 ≈ 20 > epsilon_high
    loss = group_policy_loss(logps, old, logps.detach(), torch.tensor([1.0]), loss_type='cispo', beta=0.0, epsilon_high=5.0)
    loss.sum().backward()
    torch.testing.assert_close(logps.grad, torch.tensor([[-5.0, -1.0]]))  # weight = stop-grad(min(ratio, 5)) · advantage


def test_group_policy_loss_adds_kl_term():
    lp = torch.zeros(1, 3)
    base = group_policy_loss(lp, lp, lp, torch.zeros(1), beta=0.1)
    torch.testing.assert_close(base, torch.zeros(1, 3))
    shifted = group_policy_loss(lp, lp, lp + 1.0, torch.zeros(1), beta=0.1)
    torch.testing.assert_close(shifted, 0.1 * (math.e - 2) * torch.ones(1, 3))


# ── PPO ──
def _gae_reference(rewards, values, gamma, lam):
    B, T = values.shape
    adv = torch.zeros(B, T)
    for b in range(B):
        running = 0.0
        for t in reversed(range(T)):
            nv = values[b, t + 1] if t + 1 < T else 0.0
            running = rewards[b, t] + gamma * nv - values[b, t] + gamma * lam * running
            adv[b, t] = running
    return adv


@pytest.mark.parametrize('gamma,lam', [(1.0, 0.95), (0.9, 1.0), (1.0, 1.0)])
def test_gae_matches_reference(gamma, lam):
    r, v = torch.randn(3, 6), torch.randn(3, 6)
    adv, ret = gae_advantages(r, v, gamma, lam)
    torch.testing.assert_close(adv, _gae_reference(r, v, gamma, lam))
    torch.testing.assert_close(ret, adv + v)


def test_gae_with_unit_discount_is_reward_to_go_minus_value():
    r, v = torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[0.5, 0.2, 0.1]])
    adv, _ = gae_advantages(r, v, 1.0, 1.0)
    torch.testing.assert_close(adv, torch.tensor([[1.5, 1.8, 1.9]]))


def test_masked_whiten():
    x, m = torch.randn(4, 6) * 3 + 1, (torch.rand(4, 6) > 0.3).float()
    w = masked_whiten(x, m)
    assert masked_mean(w, m).item() == pytest.approx(0, abs=1e-5)
    assert masked_mean(w ** 2, m).item() == pytest.approx(1, abs=1e-4)
    assert (w[m == 0] == 0).all()


def test_ppo_losses():
    lp, mask = torch.zeros(2, 3), torch.ones(2, 3)
    adv = torch.tensor([[1.0, -1.0, 2.0], [0.0, 1.0, 1.0]])
    loss, clipfrac, kl = ppo_policy_loss(lp, lp, lp, adv, mask, clip_epsilon=0.2, kl_coef=0.5)
    assert loss.item() == pytest.approx(-adv.mean().item()) and clipfrac.item() == 0 and kl.item() == 0
    _, clipfrac, _ = ppo_policy_loss(torch.full((2, 3), 0.5), lp, lp, adv, mask, 0.2, 0.0)
    assert clipfrac.item() == 1.0
    v, old, ret = torch.tensor([[1.0]]), torch.tensor([[0.0]]), torch.tensor([[0.0]])
    assert ppo_value_loss(v, old, ret, torch.ones(1, 1), cliprange_value=0.2).item() == pytest.approx(0.5)  # max(1², 0.2²)/2
