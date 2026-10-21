# Code structure

How the code is organised, which design patterns hold it together, and how to change it safely.
For the model architecture itself see [architecture/index.html](architecture/index.html).

## Module map

```
model/
  model_minimind.py   MiniMindConfig + the whole network (single file: it is copied as-is for transformers trust_remote_code)
  model_io.py         unwrap_model, half_state_dict, atomic_save, weight_path, load_minimind (inference loading)
  model_lora.py       LoRA adapters: apply / save / load / merge
dataset/
  lm_dataset.py       Pretrain, SFT, DPO, RLAIF and AgentRL datasets; assistant_spans() is the one label-masking rule
trainer/
  cli.py              argument groups shared by every script + config_from_args
  engine.py           run lifecycle (start_run / restore / wrap_model / run_epochs / finish_run) and SupervisedTrainer
  trainer_utils.py    distributed helpers, seeds, LR schedule, lm_checkpoint, init_model, SkipBatchSampler, reward model
  losses.py           every objective as a pure tensor function (DPO, distillation, GRPO/CISPO, GAE, PPO)
  rl_common.py        reward shaping, policy/reference/rollout setup, RL optimiser step and saving
  agent_env.py        simulated tools, multi-turn rollout (Trajectory) and agent rewards
  rollout_engine.py   RolloutEngine interface with Torch and SGLang implementations
  train_*.py          one thin file per stage: build_parser() + a small trainer class + main(argv)
eval_llm.py, scripts/ inference, serving, tool-call evaluation, conversion
tests/                pytest suite (see below)
docs/architecture/    architecture diagrams; regenerate with `python docs/architecture/build.py`
```

## Design patterns

| Pattern | Where | What it buys |
|---|---|---|
| **Template Method** | `engine.SupervisedTrainer` | The loop (LR schedule, gradient accumulation, clipping, logging, checkpointing, end-of-epoch flush) is written once. Pretrain/SFT, LoRA, DPO and distillation only implement `compute_loss(batch)` and, for LoRA, `save()`. |
| **Strategy** | `rollout_engine.RolloutEngine` | RL trainers call `rollout()` / `update_policy()` and do not know whether sampling runs in PyTorch or on an SGLang server. |
| **Factory** | `build_token_mixer`, `build_feed_forward`, `create_rollout_engine`, `config_from_args` | Choosing an implementation from configuration happens in one place. Adding a new token mixer means one new class plus one line in the factory. |
| **Dependency injection** | `GRPOTrainer`, `PPOTrainer`, `AgentTrainer`, `main(argv, reward_model=None)` | Trainers receive their models, reward model and rollout engine instead of reading module globals, so tests can pass a stub reward model and run a real step on CPU. |
| **Single source of truth** | `MiniMindConfig.weight_name`, `cli.add_*_args`, `assistant_spans` | Weight file names, CLI flags and label masking were each copied 3-25 times; each is now defined once. |
| **Pure functions** | `trainer/losses.py`, `rl_common.format_reward_terms`, sampling steps in `model_minimind.py` | Maths with no I/O can be unit-tested against hand-computed values and reference implementations. |

## Common changes

**Add a supervised training stage.** Write a dataset that yields tensors, subclass `SupervisedTrainer`, and implement
`compute_loss(batch) -> StepOutput(loss, aux_loss, metrics)`. `train_dpo.py` (about 80 lines) is the template. Compose
the parser from the `cli.add_*_args` groups with your own defaults.

**Add a model option.** Add the field to `MiniMindConfig`. If it changes the weight layout, also add it to
`MiniMindConfig.WEIGHT_SUFFIXES`, which every trainer and script uses for file names. If it should be a CLI switch,
add it to `cli.add_model_args` / `config_from_args`.

**Add a token mixer or FFN.** Implement the module with the existing `forward` signature and select it in
`build_token_mixer` / `build_feed_forward`.

**Change a reward or loss.** Edit the pure function in `losses.py`, `rl_common.py` or `agent_env.py` and update
its unit test in `tests/test_losses.py` / `tests/test_rewards.py`.

## Paths and flags

Training scripts are still run from `trainer/` and keep every flag and default they had before. Two flags were added.
Their defaults are the paths that used to be hard-coded:

- `--from_dir ../out`: where `--from_weight` is read from. PPO's critic is also initialised from here; previously it was read from `--save_dir`.
- `--checkpoint_dir ../checkpoints`: where `*_resume.pth` files are written and read.

## Tests

```bash
pip install pytest
python -m pytest                 # whole suite, about 30 s on CPU; no GPU or downloads needed
python -m pytest -m "not slow"   # unit tests only, a few seconds
```

| File | Guards |
|---|---|
| `test_model_golden.py` | Frozen tiny-model weights must reproduce the recorded logits, loss, padded logits and greedy generations for dense / MoE / KDA / AttnRes / all-on, and default-size parameter counts must not change. |
| `test_model_units.py` | Config derivation, weight names, RMSNorm, RoPE/YaRN, chunked KDA == recurrent KDA, causal short conv cache, AttnRes vs the paper formula, block boundaries, MoE routing. |
| `test_model_behaviour.py` | For every architecture: every parameter gets a gradient (DDP), incremental decoding == full forward, left padding is neutral, EOS handling, `save_pretrained` round trip, fp16 padding NaN regression. |
| `test_sampling.py`, `test_lora.py`, `test_model_io.py` | top-k / top-p / repetition penalty; LoRA identity at init, save/load, merge; weight loading. |
| `test_datasets.py` | Label masks cover exactly the assistant replies; DPO shifts and masks; RL prompt construction; resumable sampler. |
| `test_losses.py`, `test_rewards.py` | Every objective on hand-checked values or reference implementations; reward shaping, tool sandbox, ground-truth matching, multi-turn rollout masking. |
| `test_engine.py`, `test_cli.py` | Accumulation and flush, log/save cadence, checkpoint round trip and GPU-count rescaling; every script's flags and defaults against a snapshot of the pre-refactor scripts. |
| `test_trainers.py` (slow) | All supervised trainers run end to end and must match `tests/golden/train_logs.json` step by step; mid-epoch resume; GRPO, PPO and agent RL run end to end with a stub reward model. |

`tests/golden/` holds the regression fixtures. They were recorded from the code *before* the refactor, so only
regenerate them when a numeric change is intended. To regenerate, record new logs with the arguments in
`test_trainers.py` and new model outputs with the recipe in `test_model_golden.py`.
