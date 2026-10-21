import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import warnings
from dataset.lm_dataset import SFTDataset
from trainer.cli import new_parser, add_model_args, add_run_args, add_data_args, add_init_args, add_tracking_args
from trainer.train_pretrain import run_causal_lm

warnings.filterwarnings('ignore')


def build_parser():
    parser = new_parser("MiniMind Full SFT")
    add_run_args(parser, save_weight='full_sft', epochs=2, batch_size=16, learning_rate=1e-5, accumulation_steps=1, save_interval=1000)
    add_model_args(parser)
    add_data_args(parser, data_path="../dataset/sft_t2t_mini.jsonl", max_seq_len=768)
    add_init_args(parser, from_weight='pretrain')
    add_tracking_args(parser, wandb_project="MiniMind-Full-SFT")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_causal_lm(args, SFTDataset, f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}")


if __name__ == "__main__":
    main()
