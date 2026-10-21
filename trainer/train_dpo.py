import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import warnings
import torch
from torch import optim
from dataset.lm_dataset import DPODataset
from trainer.cli import new_parser, add_model_args, add_run_args, add_data_args, add_init_args, add_tracking_args, config_from_args
from trainer.engine import SupervisedTrainer, StepOutput, start_run, restore, wrap_model, make_sampler, run_epochs, finish_run
from trainer.losses import token_log_probs, dpo_loss
from trainer.trainer_utils import init_model, Logger

warnings.filterwarnings('ignore')


class DPOTrainer(SupervisedTrainer):
    """Chosen and rejected sequences run as one batch [chosen; rejected] through the policy and the frozen reference."""

    def __init__(self, *args, ref_model, beta, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model, self.beta = ref_model, beta

    def compute_loss(self, batch):
        x = torch.cat([batch['x_chosen'], batch['x_rejected']], dim=0)
        y = torch.cat([batch['y_chosen'], batch['y_rejected']], dim=0)
        mask = torch.cat([batch['mask_chosen'], batch['mask_rejected']], dim=0)
        with self.autocast_ctx:
            with torch.no_grad():
                ref_log_probs = token_log_probs(self.ref_model(x).logits, y)
            outputs = self.model(x)
            loss = dpo_loss(ref_log_probs, token_log_probs(outputs.logits, y), mask, beta=self.beta)
        return StepOutput(loss + outputs.aux_loss, outputs.aux_loss, {'dpo_loss': loss})


def build_parser():
    parser = new_parser("MiniMind DPO (Direct Preference Optimization)")
    add_run_args(parser, save_weight='dpo', epochs=1, batch_size=4, learning_rate=4e-8, accumulation_steps=1, save_interval=100,
                 lr_help="初始学习率（建议<=5e-8避免遗忘）")
    add_model_args(parser)
    add_data_args(parser, data_path="../dataset/dpo.jsonl", max_seq_len=1024)
    add_init_args(parser, from_weight='full_sft')
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    add_tracking_args(parser, wandb_project="MiniMind-DPO")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    lm_config = config_from_args(args)
    ctx = start_run(args, lm_config, args.save_weight, f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}")
    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.from_dir, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    ref_model, _ = init_model(lm_config, args.from_weight, save_dir=args.from_dir, device=args.device)  # 参考模型（冻结）
    ref_model.eval().requires_grad_(False)
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = make_sampler(train_ds)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch, start_step = restore(ctx.ckp_data, model=model, optimizer=optimizer, scaler=scaler)
    model = wrap_model(model, args, ctx.local_rank)
    trainer = DPOTrainer(args, lm_config, model, optimizer, scaler, ctx.autocast_ctx, ctx.wandb, ref_model=ref_model, beta=args.beta)
    run_epochs(args, train_ds, sampler, start_epoch, start_step, trainer.train_epoch)
    finish_run()


if __name__ == "__main__":
    main()
