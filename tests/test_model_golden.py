"""Regression: weights frozen from the pre-refactor model must reproduce the recorded outputs exactly."""
import os
import pytest
import torch
from conftest import GOLDEN, VARIANTS
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

GOLD = torch.load(os.path.join(GOLDEN, 'model_golden.pt'))


@pytest.mark.parametrize('name', list(GOLD['variants']))
def test_outputs_match_golden(name):
    v, inputs = GOLD['variants'][name], GOLD['inputs']
    model = MiniMindForCausalLM(MiniMindConfig(**GOLD['tiny'], **v['config'])).eval()
    model.load_state_dict(v['state_dict'], strict=True)  # also pins every parameter name and shape
    with torch.no_grad():
        res = model(inputs['ids'], labels=inputs['ids'])
        masked = model(inputs['ids_padded'], attention_mask=inputs['mask']).logits
        generated = model.generate(inputs['ids_padded'], attention_mask=inputs['mask'], max_new_tokens=8, do_sample=False, eos_token_id=None)
    torch.testing.assert_close(res.logits, v['logits'], rtol=0, atol=1e-5)
    torch.testing.assert_close(res.loss, v['loss'], rtol=0, atol=1e-6)
    torch.testing.assert_close(res.aux_loss, v['aux_loss'], rtol=0, atol=1e-6)
    torch.testing.assert_close(masked, v['masked_logits'], rtol=0, atol=1e-5)
    assert torch.equal(generated, v['generated'])


@pytest.mark.parametrize('name', list(VARIANTS))
def test_default_size_parameter_counts(name):
    with torch.device('meta'):
        model = MiniMindForCausalLM(MiniMindConfig(**VARIANTS[name]))
    assert sum(p.numel() for p in model.parameters()) == GOLD['default_param_counts'][name]
