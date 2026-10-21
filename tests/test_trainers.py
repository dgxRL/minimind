"""End-to-end trainer runs on CPU with tiny models and the small datasets in tests/data.

The supervised runs are compared step by step with tests/golden/train_logs.json, recorded from the trainers as they
were before the refactor (same arguments as below), so any change to data handling, loss, optimisation or schedule
shows up as a numeric difference.
"""
import json
import math
import os
import shutil

import pytest
import torch

from conftest import DATA, GOLDEN, StubRewardModel, parse_train_log
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer import engine

pytestmark = pytest.mark.slow

ALIAS = {'ce': 'ce_loss', 'distill': 'distill_loss', 'learning_rate': 'lr'}
TINY = ['--device', 'cpu', '--num_workers', '0', '--hidden_size', '64', '--num_hidden_layers', '2', '--log_interval', '1',
        '--save_interval', '1000', '--epochs', '1']


def dirs(tmp_path):
    return ['--save_dir', str(tmp_path), '--from_dir', str(tmp_path), '--checkpoint_dir', str(tmp_path / 'ckpt')]


def run(main, argv, capsys, **kwargs):
    main(argv, **kwargs)
    return parse_train_log(capsys.readouterr().out)


def assert_matches_golden(name, steps):
    golden = json.load(open(os.path.join(GOLDEN, 'train_logs.json')))[name]
    assert [s['step'] for s in steps] == [g['step'] for g in golden]
    for got, want in zip(steps, golden):
        got = {ALIAS.get(k, k): v for k, v in got.items()}
        for k, v in want.items():
            k = ALIAS.get(k, k)
            tol = 1.01e-4 if k != 'lr' else abs(v) * 1e-6 + 1e-12  # values are printed with 4 decimals
            assert abs(got[k] - v) <= tol, f'{name} step {want["step"]} {k}: {got[k]} != golden {v}'


def test_supervised_trainers_reproduce_golden_logs(tmp_path, capsys):
    from trainer import train_pretrain, train_full_sft, train_lora, train_dpo, train_distillation
    d = TINY + dirs(tmp_path)
    assert_matches_golden('pretrain', run(train_pretrain.main, d + ['--data_path', f'{DATA}/pretrain.jsonl', '--batch_size', '4',
                                                                     '--accumulation_steps', '2'], capsys))
    assert_matches_golden('pretrain_all', run(train_pretrain.main, d + ['--data_path', f'{DATA}/pretrain.jsonl', '--batch_size', '4',
                                                                         '--use_moe', '1', '--use_kda', '1', '--use_attnres', '1',
                                                                         '--save_weight', 'pretrain_all'], capsys))
    assert_matches_golden('sft', run(train_full_sft.main, d + ['--data_path', f'{DATA}/sft.jsonl', '--batch_size', '4', '--max_seq_len', '128',
                                                               '--learning_rate', '1e-3'], capsys))
    assert_matches_golden('lora', run(train_lora.main, d + ['--data_path', f'{DATA}/lora.jsonl', '--batch_size', '8', '--max_seq_len', '128',
                                                            '--lora_name', 'lora_t', '--learning_rate', '1e-3'], capsys))
    assert_matches_golden('dpo', run(train_dpo.main, d + ['--data_path', f'{DATA}/dpo.jsonl', '--batch_size', '2', '--max_seq_len', '128',
                                                          '--learning_rate', '1e-4'], capsys))
    distill_args = ['--device', 'cpu', '--num_workers', '0', '--student_hidden_size', '64', '--student_num_layers', '2',
                    '--teacher_hidden_size', '64', '--teacher_num_layers', '2', '--teacher_use_moe', '0', '--log_interval', '1',
                    '--save_interval', '1000', '--epochs', '1', '--data_path', f'{DATA}/sft.jsonl', '--batch_size', '4',
                    '--max_seq_len', '128', '--learning_rate', '1e-3', '--from_teacher_weight', 'pretrain'] + dirs(tmp_path)
    assert_matches_golden('distill', run(train_distillation.main, distill_args, capsys))

    files = set(os.listdir(tmp_path))
    assert {'pretrain_64.pth', 'pretrain_all_64_moe_kda_attnres.pth', 'full_sft_64.pth', 'lora_t_64.pth', 'dpo_64.pth', 'full_dist_64.pth'} <= files
    assert 'dpo_64_resume.pth' in os.listdir(tmp_path / 'ckpt')


def test_resume_continues_from_mid_epoch_checkpoint(tmp_path, capsys, monkeypatch):
    from trainer import train_pretrain
    argv = TINY + dirs(tmp_path) + ['--data_path', f'{DATA}/pretrain.jsonl', '--batch_size', '4', '--save_interval', '8']
    original = engine.lm_checkpoint

    def keep_step_8(*args, **kwargs):  # snapshot the resume file written at step 8
        original(*args, **kwargs)
        if kwargs.get('step') == 8: shutil.copy(tmp_path / 'ckpt' / 'pretrain_64_resume.pth', tmp_path / 'step8.pth')
    monkeypatch.setattr(engine, 'lm_checkpoint', keep_step_8)
    straight = run(train_pretrain.main, argv, capsys)
    monkeypatch.setattr(engine, 'lm_checkpoint', original)

    shutil.copy(tmp_path / 'step8.pth', tmp_path / 'ckpt' / 'pretrain_64_resume.pth')
    train_pretrain.main(argv + ['--from_resume', '1'])
    out = capsys.readouterr().out
    resumed = parse_train_log(out)
    assert '跳过前8个step' in out and [s['step'] for s in resumed] == list(range(9, 17))
    for a, b in zip(resumed, straight[8:]):
        assert a['lr'] == b['lr']                       # schedule continues where it stopped
        assert abs(a['loss'] - b['loss']) < 2e-2        # same batches; weights were restored from their fp16 copy


@pytest.fixture
def sft_weights(tmp_path):
    torch.manual_seed(0)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=64, num_hidden_layers=2))
    torch.save({k: v.half() for k, v in model.state_dict().items()}, tmp_path / 'full_sft_64.pth')
    return tmp_path


def rl_args(tmp_path, data):
    return TINY + dirs(tmp_path) + ['--data_path', f'{DATA}/{data}', '--batch_size', '2', '--max_gen_len', '8', '--max_seq_len', '128',
                                    '--learning_rate', '1e-3', '--save_interval', '4']


def assert_finite(steps, n):
    assert [s['step'] for s in steps] == list(range(1, n + 1))
    assert all(math.isfinite(v) for s in steps for v in s.values())


def test_grpo_end_to_end(sft_weights, capsys):
    from trainer import train_grpo
    rm = StubRewardModel()
    steps = run(train_grpo.main, rl_args(sft_weights, 'rlaif.jsonl') + ['--num_generations', '2'], capsys, reward_model=rm)
    assert_finite(steps, 8)
    assert len(rm.calls) == 8 * 2 * 2 and all(m and m[-1]['role'] == 'user' for m, _ in rm.calls)
    assert os.path.exists(sft_weights / 'grpo_64.pth') and os.path.exists(sft_weights / 'ckpt' / 'grpo_64_resume.pth')


def test_ppo_end_to_end(sft_weights, capsys):
    from trainer import train_ppo
    steps = run(train_ppo.main, rl_args(sft_weights, 'rlaif.jsonl') + ['--critic_learning_rate', '1e-3'], capsys, reward_model=StubRewardModel())
    assert_finite(steps, 8)
    resume = torch.load(sft_weights / 'ckpt' / 'ppo_actor_64_resume.pth')
    assert {'critic_model', 'critic_optimizer', 'critic_scheduler', 'scheduler'} <= set(resume)


def test_agent_end_to_end(sft_weights, capsys):
    from trainer import train_agent
    steps = run(train_agent.main, rl_args(sft_weights, 'agent.jsonl') + ['--num_generations', '2'], capsys, reward_model=StubRewardModel())
    assert_finite(steps, 8)
    assert all(-3.0 <= s['Reward'] <= 3.0 for s in steps)
