"""The shared training loop: LR schedule, gradient accumulation, logging/saving cadence, checkpoints and resume."""
import argparse
import os
import pytest
import torch
from torch import nn
from conftest import tiny_config, tiny_model
from trainer import trainer_utils
from trainer.engine import StepOutput, SupervisedTrainer, format_metrics, restore
from trainer.trainer_utils import get_lr, lm_checkpoint


class CountingOptimizer(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


class LinearTrainer(SupervisedTrainer):
    def compute_loss(self, batch):
        x, y = batch
        loss = ((self.model(x) - y) ** 2).mean()
        return StepOutput(loss, torch.tensor(0.0), {'mse': loss})

    def save(self, epoch, step):
        self.saved.append(step)


def make_trainer(accumulation_steps=1, log_interval=100, save_interval=100, epochs=1):
    args = argparse.Namespace(device='cpu', accumulation_steps=accumulation_steps, learning_rate=1.0, grad_clip=1.0,
                              log_interval=log_interval, save_interval=save_interval, epochs=epochs)
    model = nn.Linear(2, 1)
    opt = CountingOptimizer(model.parameters())
    trainer = LinearTrainer(args, tiny_config(), model, opt, torch.cuda.amp.GradScaler(enabled=False), torch.autocast('cpu', enabled=False))
    trainer.saved = []
    return trainer


def batches(n):
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(4, 2, generator=g), torch.randn(4, 1, generator=g)) for _ in range(n)]


def test_cosine_lr_schedule():
    assert get_lr(0, 100, 1.0) == pytest.approx(1.0)
    assert get_lr(50, 100, 1.0) == pytest.approx(0.55)
    assert get_lr(100, 100, 1.0) == pytest.approx(0.1)


@pytest.mark.parametrize('n,accum,expected', [(6, 1, 6), (6, 3, 2), (7, 3, 3), (2, 4, 1)])
def test_gradient_accumulation_and_final_flush(n, accum, expected):
    trainer = make_trainer(accumulation_steps=accum)
    trainer.train_epoch(0, batches(n), iters=n)
    assert trainer.optimizer.steps == expected  # partial window at the end of an epoch is flushed


def test_lr_is_set_every_step_and_logging_and_saving_cadence(capsys):
    trainer = make_trainer(log_interval=2, save_interval=3)
    trainer.train_epoch(0, batches(5), iters=5)
    logged = [l for l in capsys.readouterr().out.splitlines() if l.startswith('Epoch:')]
    assert [l.split(')')[0] for l in logged] == ['Epoch:[1/1](2/5', 'Epoch:[1/1](4/5', 'Epoch:[1/1](5/5']  # every 2 steps + last
    assert 'mse:' in logged[0] and 'aux_loss:' in logged[0] and 'lr:' in logged[0]
    assert trainer.saved == [3, 5]
    assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(get_lr(5, 5, 1.0))


def test_resumed_epoch_continues_step_numbering(capsys):
    trainer = make_trainer(log_interval=1)
    trainer.train_epoch(0, batches(2), iters=5, start_step=3)
    assert [l.split(')')[0] for l in capsys.readouterr().out.splitlines() if l.startswith('Epoch:')] == ['Epoch:[1/1](4/5', 'Epoch:[1/1](5/5']


def test_format_metrics():
    assert format_metrics(0, 2, 3, 10, {'loss': 1.23456, 'lr': 1e-4, 'epoch_time': 2.0}) == \
        'Epoch:[1/2](3/10), loss: 1.2346, lr: 0.00010000, epoch_time: 2.0min'


def test_checkpoint_round_trip_and_restore(tmp_path):
    model, cfg = tiny_model(), tiny_config()
    opt = torch.optim.AdamW(model.parameters())
    model(torch.randint(3, 128, (1, 4)), labels=torch.randint(3, 128, (1, 4))).loss.backward(); opt.step()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    lm_checkpoint(cfg, weight='unit', model=model, optimizer=opt, epoch=1, step=7, save_dir=str(tmp_path), scaler=scaler, note='x')
    assert sorted(os.listdir(tmp_path)) == ['unit_32.pth', 'unit_32_resume.pth']
    data = lm_checkpoint(cfg, weight='unit', save_dir=str(tmp_path))
    assert data['step'] == 7 and data['epoch'] == 1 and data['note'] == 'x' and data['world_size'] == 1
    fresh, fresh_opt = tiny_model(seed=1), torch.optim.AdamW(tiny_model(seed=1).parameters())
    assert restore(data, model=fresh, optimizer=fresh_opt, scaler=torch.cuda.amp.GradScaler(enabled=False)) == (1, 7)
    for (n, a), b in zip(model.state_dict().items(), fresh.state_dict().values()):
        torch.testing.assert_close(a.half().float(), b.float(), msg=n)
    assert restore(None, model=fresh) == (0, 0)
    assert lm_checkpoint(cfg, weight='missing', save_dir=str(tmp_path)) is None


def test_resume_rescales_step_when_gpu_count_changes(tmp_path, monkeypatch):
    model, cfg = tiny_model(), tiny_config()
    lm_checkpoint(cfg, weight='ws', model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), step=10, save_dir=str(tmp_path))
    monkeypatch.setattr(trainer_utils, 'world_size', lambda: 2)
    assert lm_checkpoint(cfg, weight='ws', save_dir=str(tmp_path))['step'] == 5
