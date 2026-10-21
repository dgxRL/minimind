"""Unit tests of the model building blocks."""
import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_config
from model.model_minimind import (AttnRes, Attention, FeedForward, KimiDeltaAttention, MiniMindConfig, MOEFeedForward, RMSNorm,
                                  ShortConvolution, apply_rotary_pos_emb, build_feed_forward, build_token_mixer, get_past_length,
                                  l2norm, precompute_freqs_cis, repeat_kv, torch_chunk_kda, torch_recurrent_kda)


# ── config ──
def test_layer_types_follow_full_attn_interval():
    assert MiniMindConfig(num_hidden_layers=8).layer_types == ['full_attention'] * 8
    assert MiniMindConfig(num_hidden_layers=8, use_kda=True).layer_types == ['kda', 'kda', 'kda', 'full_attention'] * 2
    assert MiniMindConfig(num_hidden_layers=4, use_kda=True, full_attn_interval=0).layer_types == ['kda'] * 4
    assert MiniMindConfig(num_hidden_layers=2, use_kda=True, layer_types=['full_attention', 'kda']).layer_types == ['full_attention', 'kda']


def test_attn_nope_defaults_to_use_kda():
    assert not MiniMindConfig().attn_nope and MiniMindConfig(use_kda=True).attn_nope
    assert not MiniMindConfig(use_kda=True, attn_nope=False).attn_nope


@pytest.mark.parametrize('flags,name', [({}, 'full_sft_768'), (dict(use_moe=True), 'full_sft_768_moe'),
                                        (dict(use_kda=True, use_attnres=True), 'full_sft_768_kda_attnres'),
                                        (dict(use_moe=True, use_kda=True, use_attnres=True), 'full_sft_768_moe_kda_attnres')])
def test_weight_name(flags, name):
    assert MiniMindConfig(**flags).weight_name('full_sft') == name


def test_weight_name_survives_config_round_trip(tmp_path):
    MiniMindConfig(hidden_size=512, use_kda=True).save_pretrained(tmp_path)
    assert MiniMindConfig.from_pretrained(tmp_path).weight_name('dpo') == 'dpo_512_kda'


def test_factories_pick_modules():
    cfg = tiny_config(use_kda=True, use_moe=True)
    assert isinstance(build_token_mixer(cfg, 0), KimiDeltaAttention) and isinstance(build_token_mixer(cfg, 3), Attention)
    assert isinstance(build_feed_forward(cfg), MOEFeedForward) and isinstance(build_feed_forward(tiny_config()), FeedForward)


# ── primitives ──
def test_rmsnorm_matches_formula_and_keeps_dtype():
    norm, x = RMSNorm(8, eps=1e-6), torch.randn(3, 8)
    norm.weight.data = torch.rand(8)
    torch.testing.assert_close(norm(x), norm.weight * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    assert norm(x.bfloat16()).dtype == torch.bfloat16


def test_rope_tables_and_rotation():
    cos, sin = precompute_freqs_cis(dim=16, end=64)
    assert cos.shape == sin.shape == (64, 16)
    torch.testing.assert_close(cos ** 2 + sin ** 2, torch.ones_like(cos))
    q, k = torch.randn(1, 64, 2, 16), torch.randn(1, 64, 2, 16)
    rq, rk = apply_rotary_pos_emb(q, k, cos, sin)
    torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1))  # rotation preserves length
    # attention scores depend only on the relative position: shift both by 5 positions
    same = torch.cat([q[:, :1]] * 64, 1), torch.cat([k[:, :1]] * 64, 1)
    rq, rk = apply_rotary_pos_emb(*same, cos, sin)
    torch.testing.assert_close((rq[0, 10] * rk[0, 3]).sum(-1), (rq[0, 15] * rk[0, 8]).sum(-1), rtol=1e-4, atol=1e-4)


def test_yarn_scaling_only_changes_low_frequencies():
    plain, _ = precompute_freqs_cis(dim=16, end=4096)
    scaled, _ = precompute_freqs_cis(dim=16, end=4096, rope_scaling={"original_max_position_embeddings": 2048, "factor": 16})
    torch.testing.assert_close(plain[:, 0], scaled[:, 0])  # highest frequency untouched
    assert not torch.allclose(plain[:, 7], scaled[:, 7])


def test_repeat_kv():
    x = torch.randn(2, 5, 3, 4)
    assert repeat_kv(x, 1) is x
    out = repeat_kv(x, 2)
    assert out.shape == (2, 5, 6, 4) and torch.equal(out[:, :, 0], out[:, :, 1]) and torch.equal(out[:, :, 2], x[:, :, 1])


def test_get_past_length():
    assert get_past_length(None) == 0 and get_past_length([None]) == 0
    assert get_past_length([(torch.zeros(1, 7, 2, 4), torch.zeros(1, 7, 2, 4))]) == 7
    assert get_past_length([{'seen_tokens': 11}]) == 11


# ── KDA ──
def _kda_inputs(T, B=2, H=2, K=8, V=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k = l2norm(torch.randn(B, T, H, K, generator=g)), l2norm(torch.randn(B, T, H, K, generator=g))
    v = torch.randn(B, T, H, V, generator=g)
    gate = -torch.rand(B, T, H, K, generator=g)  # log decay <= 0
    beta = torch.rand(B, T, H, generator=g)
    return q, k, v, gate, beta


@pytest.mark.parametrize('T', [1, 5, 16, 37])
def test_chunked_kda_equals_recurrence(T):
    q, k, v, g, beta = _kda_inputs(T)
    s0 = torch.randn(2, 2, 8, 8)
    o_r, s_r = torch_recurrent_kda(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    o_c, s_c = torch_chunk_kda(q, k, v, g, beta, initial_state=s0, output_final_state=True)
    torch.testing.assert_close(o_c, o_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s_c, s_r, rtol=1e-4, atol=1e-5)


def test_kda_state_carries_across_calls():
    q, k, v, g, beta = _kda_inputs(20)
    full, _ = torch_recurrent_kda(q, k, v, g, beta)
    first, state = torch_recurrent_kda(q[:, :12], k[:, :12], v[:, :12], g[:, :12], beta[:, :12], output_final_state=True)
    second, _ = torch_chunk_kda(q[:, 12:], k[:, 12:], v[:, 12:], g[:, 12:], beta[:, 12:], initial_state=state)
    torch.testing.assert_close(torch.cat([first, second], 1), full, rtol=1e-4, atol=1e-5)


def test_short_convolution_is_causal_and_cacheable():
    conv, x = ShortConvolution(6, kernel_size=4), torch.randn(2, 10, 6)
    full, _ = conv(x)
    head, cache = conv(x[:, :7])
    tail, _ = conv(x[:, 7:], cache)
    torch.testing.assert_close(torch.cat([head, tail], 1), full)
    x2 = x.clone(); x2[:, 8:] += 1  # changing the future must not change the past
    torch.testing.assert_close(conv(x2)[0][:, :8], full[:, :8])


# ── Attention Residuals ──
def test_attnres_matches_paper_formula():
    res = AttnRes(tiny_config())
    res.query.data.normal_(); res.norm.weight.data.uniform_(0.5, 1.5)
    vs = [torch.randn(2, 3, 32) for _ in range(4)]
    out = res([AttnRes.source(v, res.norm.eps) for v in vs])
    V = torch.stack(vs)
    alpha = torch.einsum('d,nbtd->nbt', res.query, res.norm(V)).softmax(0)
    torch.testing.assert_close(out, torch.einsum('nbt,nbtd->btd', alpha, V), rtol=1e-5, atol=1e-5)


def test_attnres_zero_query_is_plain_average():
    res, vs = AttnRes(tiny_config()), [torch.randn(1, 2, 32) for _ in range(3)]
    torch.testing.assert_close(res([AttnRes.source(v, 1e-6) for v in vs]), torch.stack(vs).mean(0))


def test_attnres_source_scale():
    v = torch.randn(2, 3, 32)
    _, r = AttnRes.source(v, 1e-6)
    torch.testing.assert_close(r, torch.rsqrt(v.pow(2).mean(-1) + 1e-6))


def test_attnres_block_boundaries():
    from model.model_minimind import MiniMindBlock
    close = lambda s: [MiniMindBlock(i, tiny_config(use_attnres=True, attnres_block_size=s)).attn_res_close for i in range(4)]
    assert close(1) == [[True, True]] * 4                                   # full AttnRes: every sub-layer closes a block
    assert close(2) == [[False, True]] * 4                                  # one block per layer
    assert close(4) == [[False, False], [False, True]] * 2                  # two layers per block


# ── MoE ──
def test_moe_routes_each_token_to_its_top_expert():
    cfg = tiny_config(use_moe=True)
    moe, x = MOEFeedForward(cfg).train(), torch.randn(2, 5, 32)
    y = moe(x)
    assert y.shape == x.shape and moe.aux_loss.item() > 0
    flat = x.view(-1, 32)
    top = F.softmax(moe.gate(flat), -1).argmax(-1)
    expected = torch.stack([moe.experts[e](flat[i:i + 1])[0] for i, e in enumerate(top.tolist())])  # top-1 weight normalises to 1
    torch.testing.assert_close(y.view(-1, 32), expected, rtol=1e-5, atol=1e-6)
    moe.eval(); moe(x)
    assert moe.aux_loss.item() == 0
