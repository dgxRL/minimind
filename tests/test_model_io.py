import os
import torch
from conftest import ROOT, tiny_config, tiny_model
from model.model_io import atomic_save, half_state_dict, load_minimind, unwrap_model, weight_path


def test_weight_path():
    assert weight_path('../out', 'full_sft', tiny_config(use_moe=True)) == '../out/full_sft_32_moe.pth'


def test_unwrap_model_strips_compile_wrapper():
    model = tiny_model()
    class Compiled: _orig_mod = model
    assert unwrap_model(Compiled()) is model and unwrap_model(model) is model


def test_half_state_dict_and_atomic_save(tmp_path):
    sd = half_state_dict(tiny_model())
    assert all(v.dtype == torch.float16 and v.device.type == 'cpu' for v in sd.values() if v.is_floating_point())
    atomic_save(sd, str(tmp_path / 'w.pth'))
    assert os.listdir(tmp_path) == ['w.pth']


def test_load_minimind_native_weights(tmp_path):
    model, config = tiny_model(), tiny_config()
    torch.save(model.state_dict(), tmp_path / 'w.pth')
    loaded, tokenizer = load_minimind(os.path.join(ROOT, 'model'), str(tmp_path / 'w.pth'), config)
    assert tokenizer.eos_token == '<|im_end|>'
    ids = torch.randint(3, 128, (1, 5))
    with torch.no_grad(): torch.testing.assert_close(loaded(ids).logits, model(ids).logits)
