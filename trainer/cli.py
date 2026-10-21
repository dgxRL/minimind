"""Command-line argument groups shared by the trainers and inference scripts.

Each script composes the groups it needs and passes its own defaults, so a flag is declared once instead of in
every file. `config_from_args` turns the model flags back into a MiniMindConfig.
"""
import argparse
import torch
from model.model_minimind import MiniMindConfig

FLAG = dict(type=int, choices=[0, 1])


def default_device(index=True):
    if not torch.cuda.is_available(): return "cpu"
    return "cuda:0" if index else "cuda"


def add_model_args(parser, prefix='', layers_flag='num_hidden_layers', who='', hidden_size=768, num_layers=8, use_moe=0):
    """--{prefix}hidden_size, --{prefix}<layers_flag>, --{prefix}use_moe/use_kda/use_attnres."""
    parser.add_argument(f'--{prefix}hidden_size', default=hidden_size, type=int, help=f"{who}隐藏层维度")
    parser.add_argument(f'--{prefix}{layers_flag}', default=num_layers, type=int, help=f"{who}隐藏层数量")
    parser.add_argument(f'--{prefix}use_moe', default=use_moe, **FLAG, help=f"{who}是否使用MoE架构（0=否，1=是）")
    parser.add_argument(f'--{prefix}use_kda', default=0, **FLAG, help=f"{who}是否使用KDA线性注意力混合架构（0=否，1=是）")
    parser.add_argument(f'--{prefix}use_attnres', default=0, **FLAG, help=f"{who}是否使用注意力残差Block AttnRes（0=否，1=是）")


def config_from_args(args, prefix='', layers_flag='num_hidden_layers', **extra):
    """Inverse of add_model_args; extra kwargs (e.g. max_seq_len, inference_rope_scaling) go straight to MiniMindConfig."""
    get = lambda name: getattr(args, prefix + name)
    return MiniMindConfig(hidden_size=get('hidden_size'), num_hidden_layers=get(layers_flag),
                          use_moe=bool(get('use_moe')), use_kda=bool(get('use_kda')), use_attnres=bool(get('use_attnres')), **extra)


def add_run_args(parser, *, save_weight, epochs, batch_size, learning_rate, accumulation_steps=1, log_interval=100,
                 save_interval=100, lr_help="初始学习率"):
    """Output locations, optimisation schedule and hardware."""
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    if save_weight is not None:
        parser.add_argument('--save_weight', default=save_weight, type=str, help="保存权重的前缀名")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="续训checkpoint目录")
    parser.add_argument("--epochs", type=int, default=epochs, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=learning_rate, help=lr_help)
    parser.add_argument("--device", type=str, default=default_device(), help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=accumulation_steps, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=log_interval, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=save_interval, help="模型保存间隔")


def add_data_args(parser, *, data_path, max_seq_len, max_seq_len_help="训练的最大截断长度（中文1token≈1.5~1.7字符）"):
    parser.add_argument('--max_seq_len', default=max_seq_len, type=int, help=max_seq_len_help)
    parser.add_argument("--data_path", type=str, default=data_path, help="训练数据路径")


def add_init_args(parser, *, from_weight):
    """Initial weights (--from_weight, read from --from_dir) and auto-resume."""
    if from_weight is not None:
        parser.add_argument('--from_weight', default=from_weight, type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_dir', default='../out', type=str, help="--from_weight 所在目录")
    parser.add_argument('--from_resume', default=0, **FLAG, help="是否自动检测&续训（0=否，1=是）")


def add_tracking_args(parser, *, wandb_project):
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default=wandb_project, help="wandb项目名")
    parser.add_argument("--use_compile", default=0, **FLAG, help="是否使用torch.compile加速（0=否，1=是）")


def add_rl_args(parser, *, max_gen_len, thinking_ratio, sglang_shared_path):
    """Generation length, reward model, debugging and the pluggable rollout engine."""
    parser.add_argument("--max_gen_len", type=int, default=max_gen_len, help="生成的最大长度")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=thinking_ratio, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default=sglang_shared_path, help="SGLang共享存储路径")


def add_group_policy_args(parser, *, num_generations):
    """GRPO / CISPO objective (train_grpo, train_agent)."""
    parser.add_argument("--num_generations", type=int, default=num_generations, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")


def add_inference_args(parser, *, load_from, save_dir, weight='full_sft'):
    """Weight selection for the inference scripts (eval_llm, serve_openai_api, eval_toolcall)."""
    parser.add_argument('--load_from', default=load_from, type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default=save_dir, type=str, help="模型权重目录")
    parser.add_argument('--weight', default=weight, type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")


def new_parser(description):
    return argparse.ArgumentParser(description=description)
