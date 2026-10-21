import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
try:  # Triton kernels from flash-linear-attention (pip install flash-linear-attention); pure PyTorch fallback otherwise
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
except ImportError:
    chunk_kda = fused_recurrent_kda = None

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    """Hyper-parameters of MiniMind. The three architecture switches are independent:
    use_moe (MoE FFN), use_kda (Kimi Delta Attention hybrid), use_attnres (Attention Residuals)."""
    model_type = "minimind"
    # architecture switches that change the weight layout, in the order they appear in weight file names
    WEIGHT_SUFFIXES = (("use_moe", "_moe"), ("use_kda", "_kda"), ("use_attnres", "_attnres"))

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, use_kda=False, use_attnres=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.use_kda = use_kda
        self.use_attnres = use_attnres
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        ### KDA (Kimi Delta Attention, arXiv:2510.26692) specific configs (ignored if use_kda = False)
        # every `full_attn_interval`-th layer keeps full attention, the rest are KDA (4 -> 3:1 KDA:Attn as in Kimi Linear; <=0 -> pure KDA)
        self.full_attn_interval = kwargs.get("full_attn_interval", 4)
        self.layer_types = kwargs.get("layer_types", [
            "kda" if use_kda and (self.full_attn_interval <= 0 or (i + 1) % self.full_attn_interval != 0) else "full_attention"
            for i in range(num_hidden_layers)
        ])
        self.kda_num_heads = kwargs.get("kda_num_heads", self.num_attention_heads)
        self.kda_head_dim = kwargs.get("kda_head_dim", self.head_dim)
        self.kda_conv_size = kwargs.get("kda_conv_size", 4)
        # Kimi Linear uses NoPE in full attention layers, leaving positional information to KDA
        self.attn_nope = kwargs.get("attn_nope", use_kda)
        ### Attention Residuals (arXiv:2603.15031) specific configs (ignored if use_attnres = False)
        # block size counted in sub-layers (each Transformer layer = attention + MLP = 2 sub-layers):
        # 1 -> Full AttnRes, 2 -> one block per layer (8 blocks for 8 layers, the paper's N≈8), 2*num_hidden_layers -> one block
        self.attnres_block_size = kwargs.get("attnres_block_size", 2)

    def weight_name(self, prefix: str) -> str:
        """File stem used by every trainer and script, e.g. weight_name('pretrain') -> 'pretrain_768_moe_kda'."""
        return f"{prefix}_{self.hidden_size}" + "".join(suffix for flag, suffix in self.WEIGHT_SUFFIXES if getattr(self, flag, False))

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        )

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        self.use_rope = not config.attn_nope

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)
        if self.use_rope:
            cos, sin = position_embeddings
            xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2), 
            repeat_kv(xk, self.n_rep).transpose(1, 2), 
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        
        if (
            self.flash 
            and (seq_len > 1) 
            and (not self.is_causal or past_key_value is None) 
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(xq, xk, xv, 
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)  # fp32: -1e9 pad mask overflows to -inf in fp16
            if self.is_causal: 
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)

            if attention_mask is not None: 
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv

# KDA recurrence (Kimi Linear Eq.1): S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T,  o_t = S_t^T q_t
# q, k: [B, T, H, K] (L2-normalized), v: [B, T, H, V], g = log(α): [B, T, H, K], beta: [B, T, H], state S: [B, H, K, V]
def torch_recurrent_kda(q, k, v, g, beta, initial_state=None, output_final_state=False):
    dtype, (B, T, H, K), V = v.dtype, q.shape, v.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    S = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.float()
    o = []
    for t in range(T):
        S = S * g[:, t, :, :, None].exp()
        S = S + torch.einsum('bhk,bhv->bhkv', beta[:, t, :, None] * k[:, t], v[:, t] - torch.einsum('bhk,bhkv->bhv', k[:, t], S))
        o.append(torch.einsum('bhk,bhkv->bhv', q[:, t] * K ** -0.5, S))
    return torch.stack(o, dim=1).to(dtype), (S if output_final_state else None)

# Chunkwise form (WY representation + UT transform, Kimi Linear Eq.6-9); slow & memory hungry, only used without fla kernels
def torch_chunk_kda(q, k, v, g, beta, initial_state=None, output_final_state=False, chunk_size=16):
    dtype, (B, T, H, K), V, C = v.dtype, q.shape, v.shape[-1], chunk_size
    pad = (C - T % C) % C  # padded steps have g=0, k=0: they leave the state untouched
    q, k, v, g, beta = (F.pad(x, (0, 0) * (x.dim() - 2) + (0, pad)).float().transpose(1, 2).reshape(B, H, -1, C, *x.shape[3:])
                        for x in (q, k, v, g, beta))
    q, g = q * K ** -0.5, g.cumsum(-2)
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    decay = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(~causal[..., None], float('-inf')).exp()  # [i, j] = Π_{j<s<=i} α_s
    A_kk = (torch.einsum('...ik,...ijk,...jk->...ij', k, decay, k) * beta[..., None]).masked_fill(~causal.tril(-1), 0)
    T_mat = torch.linalg.solve_triangular(A_kk + torch.eye(C, device=q.device), torch.diag_embed(beta), upper=False)
    w, u = T_mat @ (g.exp() * k), T_mat @ v
    A_qk = torch.einsum('...ik,...ijk,...jk->...ij', q, decay, k)
    S = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.float()
    o = []
    for n in range(q.shape[2]):
        g_n, g_last = g[:, :, n], g[:, :, n, -1]
        v_n = u[:, :, n] - w[:, :, n] @ S
        o.append((q[:, :, n] * g_n.exp()) @ S + A_qk[:, :, n] @ v_n)
        S = S * g_last[..., None].exp() + ((g_last.unsqueeze(-2) - g_n).exp() * k[:, :, n]).transpose(-1, -2) @ v_n
    o = torch.stack(o, dim=2).reshape(B, H, -1, V)[:, :, :T].transpose(1, 2)
    return o.to(dtype), (S if output_final_state else None)

def l2norm(x, eps=1e-6):
    return (x.float() * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + eps)).type_as(x)

class ShortConvolution(nn.Module):
    """Causal depthwise conv1d + SiLU. The cache holds the last (kernel_size - 1) inputs for incremental decoding."""
    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(dim, kernel_size).uniform_(-kernel_size ** -0.5, kernel_size ** -0.5))

    def forward(self, x, cache=None):
        if cache is None: cache = x.new_zeros(x.shape[0], self.kernel_size - 1, x.shape[-1])
        x = torch.cat([cache.to(x.dtype), x], dim=1)
        out = F.conv1d(x.transpose(1, 2), self.weight.unsqueeze(1).to(x.dtype), groups=x.shape[-1]).transpose(1, 2)
        return F.silu(out), x[:, -(self.kernel_size - 1):]

class KimiDeltaAttention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_heads, self.head_dim = config.kda_num_heads, config.kda_head_dim
        self.key_dim = self.value_dim = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        # one depthwise conv over [q|k|v] channels == three separate short convs
        self.conv1d = ShortConvolution(self.key_dim * 2 + self.value_dim, config.kda_conv_size)
        # fine-grained (per-channel) decay α = exp(-exp(A_log) * softplus(W_up W_down x + dt_bias)), low rank = head_dim
        self.f_proj = nn.Sequential(nn.Linear(config.hidden_size, self.head_dim, bias=False), nn.Linear(self.head_dim, self.key_dim, bias=False))
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.num_heads).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(self.key_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        # low-rank sigmoid output gate on top of head-wise RMSNorm (Eq.10)
        self.g_proj = nn.Sequential(nn.Linear(config.hidden_size, self.head_dim, bias=False), nn.Linear(self.head_dim, self.value_dim, bias=True))
        self.o_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, position_embeddings=None, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        mask = attention_mask[:, -seq_len:].bool() if attention_mask is not None else None
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        if mask is not None: qkv = qkv.masked_fill(~mask[..., None], 0)  # padding must not leak into the conv window
        qkv, conv_state = self.conv1d(qkv, past_key_value['conv_state'] if past_key_value is not None else None)
        q, k, v = (t.reshape(bsz, seq_len, self.num_heads, self.head_dim) for t in qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1))

        g = -self.A_log.float().exp()[:, None] * F.softplus(
            self.f_proj(x).float().view(bsz, seq_len, self.num_heads, self.head_dim) + self.dt_bias.float().view(self.num_heads, self.head_dim))
        beta = self.b_proj(x).float().sigmoid()
        if mask is not None:  # padding leaves the recurrent state untouched: no decay (g=0), no write (β=0)
            g, beta = g.masked_fill(~mask[..., None, None], 0), beta.masked_fill(~mask[..., None], 0)

        initial_state = past_key_value['recurrent_state'] if past_key_value is not None else None
        if chunk_kda is not None and x.is_cuda:
            kda_fn = fused_recurrent_kda if (seq_len <= 64 and not torch.is_grad_enabled()) else chunk_kda
            if kda_fn is fused_recurrent_kda and self.head_dim & (self.head_dim - 1):
                # fused_recurrent_kda (fla 0.5.2) loads g without a mask over next_power_of_2(K) lanes; with K=96 the last head
                # reads past the end of g and stray values give exp(g)=inf -> NaN. A zero tail keeps those lanes harmless.
                g = F.pad(g.flatten(), (0, self.head_dim))[:g.numel()].view_as(g)
            o, recurrent_state = kda_fn(q, k, v, g, beta.to(q.dtype), initial_state=initial_state,
                                        output_final_state=use_cache, use_qk_l2norm_in_kernel=True)
        else:
            kda_fn = torch_recurrent_kda if seq_len <= 16 else torch_chunk_kda
            o, recurrent_state = kda_fn(l2norm(q), l2norm(k), v, g, beta, initial_state=initial_state, output_final_state=use_cache)

        o = self.o_norm(o) * torch.sigmoid(self.g_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim))
        output = self.resid_dropout(self.o_proj(o.reshape(bsz, seq_len, -1)))
        past_kv = {
            'conv_state': conv_state,
            'recurrent_state': recurrent_state,
            'seen_tokens': (past_key_value['seen_tokens'] if past_key_value is not None else 0) + seq_len
        } if use_cache else None
        return output, past_kv

def get_past_length(past_key_values):
    """Attention layers cache (k, v) tensors; KDA layers cache a fixed-size state dict that counts seen tokens."""
    past = past_key_values[0] if past_key_values else None
    if past is None: return 0
    return past['seen_tokens'] if isinstance(past, dict) else past[0].shape[1]

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(config, intermediate_size=config.moe_intermediate_size) 
                for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)

        if self.config.norm_topk_prob: 
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class AttnRes(nn.Module):
    """Attention Residuals: softmax attention over depth, h = sum_i softmax_i(w^T RMSNorm(v_i)) * v_i"""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # pseudo-query must start at zero -> uniform weights over sources (plain Parameter so _init_weights leaves it alone)
        self.query = nn.Parameter(torch.zeros(config.hidden_size))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @staticmethod
    def source(v, eps):
        # a depth source is (v, 1/rms(v)); the scale is shared by every later layer, so completed blocks compute it once
        return v, torch.rsqrt(torch.linalg.vector_norm(v, dim=-1, dtype=torch.float32).pow(2) / v.shape[-1] + eps)

    def forward(self, sources):
        # w^T (g * v / rms(v)) == ((w * g)^T v) / rms(v): fold the RMSNorm weight into the query, no normalized copies
        # always computed, even for a single source, so every parameter gets a gradient (DDP)
        q = self.query * self.norm.weight
        logits = torch.stack([(v @ q.to(v.dtype)).float() * r for v, r in sources])  # [N, B, T]
        alpha = logits.softmax(0)
        return sum(a.to(v.dtype).unsqueeze(-1) * v for a, (v, _) in zip(alpha, sources))

def build_token_mixer(config: MiniMindConfig, layer_id: int) -> nn.Module:
    """Factory for the sequence-mixing sub-layer; both choices share forward(x, position_embeddings, past_key_value, use_cache, attention_mask)."""
    return KimiDeltaAttention(config) if config.layer_types[layer_id] == "kda" else Attention(config)

def build_feed_forward(config: MiniMindConfig) -> nn.Module:
    """Factory for the channel-mixing sub-layer."""
    return MOEFeedForward(config) if config.use_moe else FeedForward(config)

class MiniMindBlock(nn.Module):
    """Pre-norm Transformer layer: token mixer + FFN, joined either by plain residual adds or by Attention Residuals."""
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = build_token_mixer(config, layer_id)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = build_feed_forward(config)
        if config.use_attnres:
            self.attn_res, self.mlp_res = AttnRes(config), AttnRes(config)
            # close the current block after the attention / MLP sub-layer (sub-layer index 2*layer_id, 2*layer_id+1)
            self.attn_res_close = [(2 * layer_id + j + 1) % config.attnres_block_size == 0 for j in range(2)]

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None, attn_res_state=None):
        if attn_res_state is not None:
            return self.forward_attn_res(attn_res_state, position_embeddings, past_key_value, use_cache, attention_mask)
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), 
            position_embeddings,
            past_key_value, 
            use_cache, 
            attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

    def forward_attn_res(self, attn_res_state, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # blocks: completed block sources [b_0 = embedding, b_1, ...] as (v, 1/rms); partial: running sum of the current block (None at its start)
        blocks, partial = attn_res_state
        sublayers = [
            (self.attn_res, lambda h: self.self_attn(self.input_layernorm(h), position_embeddings, past_key_value, use_cache, attention_mask)),
            (self.mlp_res, lambda h: (self.mlp(self.post_attention_layernorm(h)), None)),
        ]
        present_key_value = None
        for (res, fn), close in zip(sublayers, self.attn_res_close):
            out, present = fn(res(blocks if partial is None else blocks + [AttnRes.source(partial, res.norm.eps)]))
            present_key_value = present if present is not None else present_key_value
            partial = out if partial is None else partial + out
            if close: blocks, partial = blocks + [AttnRes.source(partial, res.norm.eps)], None
        return (blocks, partial), present_key_value

class MiniMindModel(nn.Module):
    """Embedding -> N blocks -> final norm. Returns (hidden_states, per-layer cache entries, summed MoE aux loss)."""
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_attnres: self.final_attn_res = AttnRes(config)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, 
            end=config.max_position_embeddings, 
            rope_base=config.rope_theta, 
            rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, 'layers'): past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = get_past_length(past_key_values)
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim, 
                end=self.config.max_position_embeddings, 
                rope_base=self.config.rope_theta, 
                rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)

        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])

        presents = []
        attn_res_state = ([AttnRes.source(hidden_states, self.config.rms_norm_eps)], None) if self.config.use_attnres else None
        for layer, past_key_value in zip(self.layers, past_key_values):
            out, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
                attn_res_state=attn_res_state
            )
            if attn_res_state is None: hidden_states = out
            else: attn_res_state = out
            presents.append(present)
        if attn_res_state is not None:
            # the output layer aggregates the embedding and all block representations
            blocks, partial = attn_res_state
            hidden_states = self.final_attn_res(blocks if partial is None else blocks + [AttnRes.source(partial, self.config.rms_norm_eps)])
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

# ── sampling: each step edits the next-token logits in place, then sample_next_token picks a token ──
def apply_repetition_penalty(logits, input_ids, penalty):
    """CTRL-style penalty on every token already present in the row: positive logits / penalty, negative * penalty."""
    for i in range(input_ids.shape[0]):
        seen = torch.unique(input_ids[i])
        score = logits[i, seen]
        logits[i, seen] = torch.where(score > 0, score / penalty, score * penalty)
    return logits

def apply_top_k(logits, top_k):
    """Keep only the top_k largest logits per row."""
    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
    return logits

def apply_top_p(logits, top_p):
    """Nucleus filtering: keep the smallest set of tokens whose probability mass exceeds top_p (always at least one)."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
    mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
    logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
    return logits

def sample_next_token(logits, input_ids, do_sample=True, top_k=50, top_p=0.85, repetition_penalty=1.0):
    """[B, V] temperature-scaled logits -> [B, 1] next token ids."""
    if repetition_penalty != 1.0: logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
    if top_k > 0: logits = apply_top_k(logits, top_k)
    if top_p < 1.0: logits = apply_top_p(logits, top_p)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(
            loss=loss, 
            aux_loss=aux_loss, 
            logits=logits, 
            past_key_values=past_key_values, 
            hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None

        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if streamer: streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = get_past_length(past_key_values)
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            next_token = sample_next_token(logits, input_ids, do_sample, top_k, top_p, repetition_penalty)
            if eos_token_id is not None: 
                next_token = torch.where(
                    finished.unsqueeze(-1), 
                    next_token.new_full((next_token.shape[0], 1), eos_token_id), 
                    next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids