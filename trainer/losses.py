"""Pure tensor functions for every training objective (no model, no I/O) — each one is unit-tested in tests/."""
import torch
import torch.nn.functional as F


# ───────────────────────────── shared ─────────────────────────────
def token_log_probs(logits, labels):
    """log p(labels) per position: logits [B, T, V], labels [B, T] -> [B, T]."""
    return torch.gather(F.log_softmax(logits, dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def masked_mean(values, mask):
    """Mean of values over mask == 1 (whole tensor); 0 when the mask is empty."""
    return (values * mask).sum() / mask.sum().clamp(min=1)


def masked_row_mean(values, mask):
    """Per-row masked mean [B, T] -> [B]."""
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


# ───────────────────────────── DPO / distillation ─────────────────────────────
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """Direct Preference Optimization. Rows [:B/2] are chosen responses, rows [B/2:] the matching rejected ones."""
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    half = ref_log_probs.shape[0] // 2
    pi_logratios = policy_log_probs[:half] - policy_log_probs[half:]
    ref_logratios = ref_log_probs[:half] - ref_log_probs[half:]
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """T² · KL(teacher_T || student_T) with temperature-softened distributions."""
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return (temperature ** 2) * F.kl_div(student_log_probs, teacher_probs, reduction=reduction)


# ───────────────────────────── group policy optimisation (GRPO / CISPO) ─────────────────────────────
def group_advantages(rewards, num_generations, eps=1e-4):
    """Normalise each reward within its group of num_generations samples of the same prompt.
    Returns (advantages [N], grouped rewards [N / num_generations, num_generations])."""
    grouped = rewards.view(-1, num_generations)
    mean_r = grouped.mean(dim=1).repeat_interleave(num_generations)
    std_r = grouped.std(dim=1, unbiased=False).repeat_interleave(num_generations)
    return (rewards - mean_r) / (std_r + eps), grouped


def truncate_after_eos(mask, is_eos):
    """Zero mask positions after the first EOS of each row (the EOS itself stays). Rows without EOS are unchanged."""
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=mask.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    keep = torch.arange(is_eos.size(1), device=mask.device).unsqueeze(0) <= eos_idx.unsqueeze(1)
    return (mask.bool() & keep).to(mask.dtype) if mask.dtype in (torch.bool, torch.int32, torch.int64) else mask * keep.to(mask.dtype)


def kl_k3(ref_logps, logps):
    """Per-token k3 estimator of KL(policy || ref): exp(r) - r - 1 with r = ref - policy (always >= 0)."""
    log_ratio = ref_logps - logps
    return torch.exp(log_ratio) - log_ratio - 1


def group_policy_loss(logps, old_logps, ref_logps, advantages, loss_type="cispo", beta=0.1, epsilon=0.2, epsilon_high=5.0):
    """Per-token loss [N, T]. 'grpo': PPO-clipped ratio · advantage; 'cispo': stop-grad clipped ratio weighting the
    log-prob (REINFORCE-style, keeps gradients on clipped tokens). Both add beta · k3 KL to the reference."""
    per_token_kl = kl_k3(ref_logps, logps)
    ratio = torch.exp(logps - old_logps)
    adv = advantages.unsqueeze(1)
    if loss_type == "cispo":
        clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()
        return -(clamped_ratio * adv * logps - beta * per_token_kl)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return -(torch.min(ratio * adv, clipped_ratio * adv) - beta * per_token_kl)


# ───────────────────────────── PPO ─────────────────────────────
def gae_advantages(token_rewards, values, gamma, lam):
    """Generalised Advantage Estimation over the response tokens. Returns (advantages, returns), both [B, R]."""
    gen_len = values.size(1)
    lastgaelam = torch.zeros(values.size(0), device=values.device)
    advs_rev = []
    for t in reversed(range(gen_len)):
        next_value = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advs_rev.append(lastgaelam)
    advantages = torch.stack(advs_rev[::-1], dim=1)
    return advantages, advantages + values


def masked_whiten(values, mask, eps=1e-8):
    """(x - mean) / std over masked positions; masked-out positions become 0."""
    mean = masked_mean(values, mask)
    var = masked_mean((values - mean) ** 2, mask)
    return (values - mean) * torch.rsqrt(var + eps) * mask


def ppo_policy_loss(logps, old_logps, ref_logps, advantages, mask, clip_epsilon, kl_coef):
    """Clipped surrogate + kl_coef · k3 KL to the reference. Returns (loss, clipfrac, kl_ref)."""
    ratio = torch.exp(logps - old_logps)
    clipfrac = masked_mean(((ratio - 1.0).abs() > clip_epsilon).float(), mask)
    kl_ref = masked_mean(kl_k3(ref_logps, logps), mask)
    surrogate = torch.max(-advantages * ratio, -advantages * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon))
    return masked_mean(surrogate, mask) + kl_coef * kl_ref, clipfrac, kl_ref


def ppo_value_loss(values, old_values, returns, mask, cliprange_value):
    """0.5 · max(unclipped, clipped) squared error of the critic."""
    clipped = torch.clamp(values, old_values - cliprange_value, old_values + cliprange_value)
    return 0.5 * masked_mean(torch.max((values - returns) ** 2, (clipped - returns) ** 2), mask)
