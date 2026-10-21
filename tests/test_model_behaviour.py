"""End-to-end model properties for every architecture combination (dense / MoE / KDA / AttnRes)."""
import torch
from conftest import tiny_config, tiny_model
from model.model_minimind import MiniMindForCausalLM


def _ids(B=2, T=24, seed=1):
    return torch.randint(3, 128, (B, T), generator=torch.Generator().manual_seed(seed))


def _randomise_attnres(model):
    for n, p in model.named_parameters():
        if n.endswith('.query'): p.data.normal_(0, 0.5)


def test_every_parameter_gets_a_gradient(variant):
    model = tiny_model(**variant[1]).train()
    res = model(_ids(), labels=_ids())
    (res.loss + res.aux_loss).backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []  # DDP requires every parameter to be used
    assert torch.isfinite(res.loss)


def test_attnres_queries_start_at_zero():
    model = tiny_model(use_attnres=True)
    queries = [p for n, p in model.named_parameters() if n.endswith('.query')]
    assert len(queries) == 2 * 4 + 1 and all(torch.count_nonzero(q) == 0 for q in queries)


def test_incremental_decoding_matches_full_forward(variant):
    model = tiny_model(**variant[1]).eval(); _randomise_attnres(model)
    ids = _ids()
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :16], use_cache=True)
        logits, cache = [out.logits], out.past_key_values
        for t in range(16, 24):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            logits.append(out.logits); cache = out.past_key_values
    torch.testing.assert_close(torch.cat(logits, 1), full, rtol=1e-4, atol=1e-4)


def test_left_padding_does_not_change_real_tokens(variant):
    model = tiny_model(**variant[1]).eval(); _randomise_attnres(model)
    ids = _ids(B=1, T=20)
    padded = torch.cat([torch.zeros(1, 5, dtype=torch.long), ids], 1)
    mask = torch.cat([torch.zeros(1, 5, dtype=torch.long), torch.ones(1, 20, dtype=torch.long)], 1)
    with torch.no_grad():
        torch.testing.assert_close(model(padded, attention_mask=mask).logits[:, 5:], model(ids).logits, rtol=1e-4, atol=1e-4)


def test_generate_stops_rows_at_eos_and_pads_with_eos():
    model = tiny_model().eval()
    ids = _ids(B=2, T=6)
    with torch.no_grad():
        greedy = model.generate(ids, max_new_tokens=5, do_sample=False, eos_token_id=None)
        eos = greedy[0, 6].item()  # make row 0's first generated token the EOS
        out = model.generate(ids, max_new_tokens=5, do_sample=False, eos_token_id=eos)
    assert out[0, 6:].eq(eos).all()
    assert out.shape[1] <= 11


def test_generate_num_return_sequences_and_streamer():
    class Collect:
        def __init__(self): self.chunks, self.ended = [], False
        def put(self, t): self.chunks.append(t)
        def end(self): self.ended = True
    model, streamer = tiny_model().eval(), Collect()
    out = model.generate(_ids(B=1, T=4), max_new_tokens=3, do_sample=True, num_return_sequences=2, eos_token_id=None, streamer=streamer)
    assert out.shape == (2, 7) and streamer.ended and len(streamer.chunks) == 1 + 3


def test_save_pretrained_round_trip(tmp_path, variant):
    model = tiny_model(**variant[1]).eval(); _randomise_attnres(model)
    model.save_pretrained(tmp_path, safe_serialization=False)
    loaded = MiniMindForCausalLM.from_pretrained(tmp_path).eval()
    assert loaded.config.weight_name('x') == model.config.weight_name('x')
    with torch.no_grad():
        torch.testing.assert_close(loaded(_ids()).logits, model(_ids()).logits)


def test_tied_embeddings():
    model = tiny_model()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert MiniMindForCausalLM(tiny_config(tie_word_embeddings=False)).lm_head.weight is not None


def test_fp16_eager_attention_with_padding_has_no_nan():
    """Regression: the -1e9 padding mask used to overflow to -inf in fp16 and turn fully padded rows into NaN."""
    model = tiny_model().eval()
    ids = _ids(B=2, T=8)
    mask = torch.ones_like(ids); mask[0, :3] = 0
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.float16):
        assert torch.isfinite(model(ids, attention_mask=mask).logits).all()
