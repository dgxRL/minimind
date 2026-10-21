import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import warnings
import torch
from torch import optim
from dataset.lm_dataset import SFTDataset
from model.model_io import weight_path
from model.model_lora import save_lora, apply_lora
from trainer.cli import new_parser, add_model_args, add_run_args, add_data_args, add_init_args, add_tracking_args, config_from_args
from trainer.engine import start_run, restore, wrap_model, make_sampler, run_epochs, finish_run
from trainer.train_pretrain import CausalLMTrainer
from trainer.trainer_utils import init_model, lm_checkpoint, Logger

warnings.filterwarnings('ignore')


class LoRATrainer(CausalLMTrainer):
    """Causal LM loss on a frozen base; only the adapters are clipped, optimised and saved."""

    def save(self, epoch, step):
        args = self.args
        save_lora(self.model, weight_path(args.save_dir, args.lora_name, self.lm_config))  # LoRA只保存LoRA权重
        lm_checkpoint(self.lm_config, weight=args.lora_name, model=self.model, optimizer=self.optimizer, scaler=self.scaler,
                      epoch=epoch, step=step, wandb=self.wandb, save_dir=args.checkpoint_dir)


def freeze_all_but_lora(model):
    """Freeze base weights; return the LoRA parameters (the only trainable ones)."""
    total = sum(p.numel() for p in model.parameters())
    lora_params = []
    for name, param in model.named_parameters():
        param.requires_grad = 'lora' in name
        if param.requires_grad: lora_params.append(param)
    count = sum(p.numel() for p in lora_params)
    Logger(f"LLM 总参数量: {total / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {count / total * 100:.2f}%")
    return lora_params


def build_parser():
    parser = new_parser("MiniMind LoRA Fine-tuning")
    add_run_args(parser, save_weight=None, epochs=10, batch_size=32, learning_rate=1e-4, accumulation_steps=1, log_interval=10, save_interval=1000)
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA权重名称(如lora_identity/lora_medical等)")
    add_model_args(parser)
    add_data_args(parser, data_path="../dataset/lora_medical.jsonl", max_seq_len=340)
    add_init_args(parser, from_weight='full_sft')
    add_tracking_args(parser, wandb_project="MiniMind-LoRA")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    lm_config = config_from_args(args)
    ctx = start_run(args, lm_config, args.lora_name, f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}")
    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.from_dir, device=args.device)
    apply_lora(model)
    lora_params = freeze_all_but_lora(model)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = make_sampler(train_ds)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    start_epoch, start_step = restore(ctx.ckp_data, strict=False, model=model, optimizer=optimizer, scaler=scaler)
    if args.use_compile == 1:
        Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
    model = wrap_model(model, args, ctx.local_rank, compile_ok=False)
    trainer = LoRATrainer(args, lm_config, model, optimizer, scaler, ctx.autocast_ctx, ctx.wandb, clip_params=lora_params)
    run_epochs(args, train_ds, sampler, start_epoch, start_step, trainer.train_epoch)
    finish_run()


if __name__ == "__main__":
    main()
