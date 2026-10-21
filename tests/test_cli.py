"""Every script keeps its flags and defaults (snapshot taken from the pre-refactor scripts), and config_from_args round-trips."""
import importlib
import json
import os
import pytest
from conftest import GOLDEN
from trainer.cli import add_model_args, config_from_args, new_parser

GOLD = json.load(open(os.path.join(GOLDEN, 'cli_defaults.json')))
NEW_FLAGS = {'checkpoint_dir': '../checkpoints', 'from_dir': '../out'}  # added by the refactor; defaults = old hard-coded paths
SCRIPTS = {'train_pretrain': 'trainer.train_pretrain', 'train_full_sft': 'trainer.train_full_sft', 'train_lora': 'trainer.train_lora',
           'train_dpo': 'trainer.train_dpo', 'train_distillation': 'trainer.train_distillation', 'train_grpo': 'trainer.train_grpo',
           'train_ppo': 'trainer.train_ppo', 'train_agent': 'trainer.train_agent', 'eval_llm': 'eval_llm',
           'eval_toolcall': 'scripts.eval_toolcall', 'serve_openai_api': 'scripts.serve_openai_api'}


@pytest.mark.parametrize('script', list(SCRIPTS))
def test_flags_and_defaults_unchanged(script):
    if script == 'serve_openai_api': pytest.importorskip('fastapi'); pytest.importorskip('uvicorn')
    parser = importlib.import_module(SCRIPTS[script]).build_parser()
    defaults = {a.dest: a.default for a in parser._actions if a.dest != 'help'}
    expected = {k: v for k, v in GOLD[script].items() if k != 'device'}  # device depends on whether CUDA is visible
    actual = {k: v for k, v in defaults.items() if k != 'device' and k not in NEW_FLAGS}
    assert actual == expected
    for k in set(defaults) & set(NEW_FLAGS): assert defaults[k] == NEW_FLAGS[k]
    assert defaults['device'] in ('cpu', 'cuda', 'cuda:0')


def test_config_from_args_with_prefix():
    parser = new_parser('t')
    add_model_args(parser, prefix='student_', layers_flag='num_layers', who='学生模型')
    args = parser.parse_args(['--student_hidden_size', '256', '--student_num_layers', '3', '--student_use_kda', '1'])
    cfg = config_from_args(args, prefix='student_', layers_flag='num_layers', max_seq_len=99)
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.use_kda, cfg.use_moe, cfg.max_seq_len) == (256, 3, True, False, 99)
    assert cfg.weight_name('full_dist') == 'full_dist_256_kda'
