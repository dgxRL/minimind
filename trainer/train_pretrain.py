import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import warnings
import torch
from torch import optim
from dataset.lm_dataset import PretrainDataset
from trainer.cli import new_parser, add_model_args, add_run_args, add_data_args, add_init_args, add_tracking_args, config_from_args
from trainer.engine import SupervisedTrainer, StepOutput, start_run, restore, wrap_model, make_sampler, run_epochs, finish_run
from trainer.trainer_utils import init_model

warnings.filterwarnings('ignore')


class CausalLMTrainer(SupervisedTrainer):
    """Next-token cross entropy (+ MoE aux loss). Labels are -100 where the loss is masked."""

    def compute_loss(self, batch):
        input_ids, labels = batch
        with self.autocast_ctx:
            res = self.model(input_ids, labels=labels)
        return StepOutput(res.loss + res.aux_loss, res.aux_loss, {'logits_loss': res.loss})


def build_parser():
    parser = new_parser("MiniMind Pretraining")
    add_run_args(parser, save_weight='pretrain', epochs=2, batch_size=32, learning_rate=5e-4, accumulation_steps=8, save_interval=1000)
    add_model_args(parser)
    add_data_args(parser, data_path="../dataset/pretrain_t2t_mini.jsonl", max_seq_len=340)
    add_init_args(parser, from_weight='none')
    add_tracking_args(parser, wandb_project="MiniMind-Pretrain")
    return parser


def run_causal_lm(args, dataset_cls, run_name):
    """Shared by pretraining and full SFT: they differ only in the dataset."""
    lm_config = config_from_args(args)
    ctx = start_run(args, lm_config, args.save_weight, run_name)
    model, tokenizer = init_model(lm_config, args.from_weight, save_dir=args.from_dir, device=args.device)
    train_ds = dataset_cls(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = make_sampler(train_ds)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch, start_step = restore(ctx.ckp_data, model=model, optimizer=optimizer, scaler=scaler)
    model = wrap_model(model, args, ctx.local_rank)
    trainer = CausalLMTrainer(args, lm_config, model, optimizer, scaler, ctx.autocast_ctx, ctx.wandb)
    run_epochs(args, train_ds, sampler, start_epoch, start_step, trainer.train_epoch)
    finish_run()


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_causal_lm(args, PretrainDataset, f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}")


if __name__ == "__main__":
    main()
