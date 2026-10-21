"""Shared fixtures. Everything runs on CPU in fp32 so results are exact and need no GPU."""
import os
import re
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
TESTS = os.path.join(ROOT, 'tests')
DATA = os.path.join(TESTS, 'data')
GOLDEN = os.path.join(TESTS, 'golden')

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

# a few thousand parameters: every architecture switch, fast on CPU
TINY = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=128)
VARIANTS = {
    'dense': {}, 'moe': dict(use_moe=True), 'kda': dict(use_kda=True), 'attnres': dict(use_attnres=True),
    'moe_kda_attnres': dict(use_moe=True, use_kda=True, use_attnres=True),
}


def tiny_config(**overrides):
    return MiniMindConfig(**{**TINY, **overrides})


def tiny_model(seed=0, **overrides):
    torch.manual_seed(seed)
    return MiniMindForCausalLM(tiny_config(**overrides))


@pytest.fixture(params=list(VARIANTS), ids=list(VARIANTS))
def variant(request):
    """(name, config kwargs) for each architecture combination."""
    return request.param, VARIANTS[request.param]


@pytest.fixture(scope='session')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))


def parse_train_log(text):
    """Per-step metrics from trainer stdout lines 'Epoch:[e/E](step/iters), key: value, ...' (timing fields dropped)."""
    steps = []
    for line in text.splitlines():
        m = re.match(r'Epoch:\[(\d+)/\d+\]\((\d+)/(\d+)\),\s*(.*)', line.strip())
        if not m: continue
        metrics = {k.strip(): float(v) for k, v in re.findall(r'([A-Za-z_][A-Za-z_ ]*?):\s*(-?\d+(?:\.\d+)?)', m.group(4))}
        metrics.pop('epoch_time', None)
        steps.append({'step': int(m.group(2)), **metrics})
    return steps


class StubRewardModel:
    """Deterministic stand-in for the 1.8B reward model; records what it was asked to score."""
    def __init__(self):
        self.calls = []

    def get_score(self, messages, response):
        self.calls.append((messages, response))
        return (len(response) % 7) / 7 - 0.5
