import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import warnings
import torch
import torch.nn.functional as F
from torch import optim
from dataset.lm_dataset import SFTDataset
from trainer.cli import new_parser, add_model_args, add_run_args, add_data_args, add_init_args, add_tracking_args, config_from_args
from trainer.engine import SupervisedTrainer, StepOutput, start_run, restore, wrap_model, make_sampler, run_epochs, finish_run
from trainer.losses import distillation_loss
from trainer.trainer_utils import init_model, Logger

warnings.filterwarnings('ignore')


class DistillationTrainer(SupervisedTrainer):
    """loss = alpha · CE(labels) + (1 - alpha) · T²·KL(teacher || student), both over the supervised (label != -100) tokens."""

    def __init__(self, *args, teacher_model, alpha, temperature, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model, self.alpha, self.temperature = teacher_model, alpha, temperature
        if teacher_model is not None:
            teacher_model.eval()
            teacher_model.requires_grad_(False)

    def compute_loss(self, batch):
        input_ids, labels = batch
        loss_mask_flat = (labels[..., 1:] != -100).float().view(-1)
        with self.autocast_ctx:
            res = self.model(input_ids)
            student_logits = res.logits[..., :-1, :].contiguous()
        vocab = student_logits.size(-1)

        ce = F.cross_entropy(student_logits.view(-1, vocab), labels[..., 1:].contiguous().view(-1), ignore_index=-100, reduction='none')
        ce_loss = torch.sum(ce * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        aux_loss = res.aux_loss if self.lm_config.use_moe else torch.tensor(0.0)

        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(input_ids).logits[..., :-1, :].contiguous()[..., :vocab]
            selected = loss_mask_flat == 1
            distill = distillation_loss(student_logits.view(-1, vocab)[selected], teacher_logits.view(-1, vocab)[selected], temperature=self.temperature)
        else:
            distill = torch.tensor(0.0, device=self.args.device)

        ce_total = ce_loss + res.aux_loss if self.lm_config.use_moe else ce_loss
        loss = self.alpha * ce_total + (1 - self.alpha) * distill
        return StepOutput(loss, aux_loss, {'ce_loss': ce_loss, 'distill_loss': distill})


def build_parser():
    # 模拟用moe模型蒸馏dense模型，也可以用更大teacher_hidden_size模型蒸馏更小student_hidden_size的
    parser = new_parser("MiniMind Knowledge Distillation")
    add_run_args(parser, save_weight='full_dist', epochs=6, batch_size=32, learning_rate=5e-6, accumulation_steps=1, save_interval=100)
    add_data_args(parser, data_path="../dataset/sft_t2t_mini.jsonl", max_seq_len=340)
    add_model_args(parser, prefix='student_', layers_flag='num_layers', who='学生模型')
    add_model_args(parser, prefix='teacher_', layers_flag='num_layers', who='教师模型', use_moe=1)
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    add_init_args(parser, from_weight=None)
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    add_tracking_args(parser, wandb_project="MiniMind-Distillation")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    student_config = config_from_args(args, prefix='student_', layers_flag='num_layers')
    teacher_config = config_from_args(args, prefix='teacher_', layers_flag='num_layers')
    ctx = start_run(args, student_config, args.save_weight,
                    f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}")
    model, tokenizer = init_model(student_config, args.from_student_weight, save_dir=args.from_dir, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    teacher_model, _ = init_model(teacher_config, args.from_teacher_weight, save_dir=args.from_dir, device=args.device)
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = make_sampler(train_ds)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch, start_step = restore(ctx.ckp_data, model=model, optimizer=optimizer, scaler=scaler)
    model = wrap_model(model, args, ctx.local_rank)
    trainer = DistillationTrainer(args, student_config, model, optimizer, scaler, ctx.autocast_ctx, ctx.wandb,
                                  teacher_model=teacher_model, alpha=args.alpha, temperature=args.temperature)
    run_epochs(args, train_ds, sampler, start_epoch, start_step, trainer.train_epoch)
    finish_run()


if __name__ == "__main__":
    main()
