import argparse
import json
import os
import time
from typing import Optional, Iterable
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cs336_basics.transformer import Transformer_LM
from cs336_basics.optimizer import AdamW, learning_rate_schedule, gradient_clipping
from cs336_basics.dataloader import data_loading
from cs336_basics.loss import cross_entropy
from cs336_basics.utils import save_checkpoint, load_checkpoint

# 可选导入 wandb（若未安装则跳过）
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore


def tokenize_text_to_binary(
    text_path: str,
    output_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str] = ("<|endoftext|>",),
    dtype: np.dtype = np.int32,
) -> int:
    from cs336_basics.optimized_tokenizer import UltraFastTokenizer

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tokenizer = UltraFastTokenizer.from_files(vocab_path, merges_path, list(special_tokens))
    
    with open(text_path, "r", encoding="utf-8") as fin, open(output_path, "wb") as fout:
        tokens: Iterable[int] = tokenizer.encode_iterable(fin)
        
        # 使用tqdm显示进度条
        with tqdm(desc="标记化进度", unit="tokens", unit_scale=True) as pbar:
            for tok in tokens:
                # 将int转换为指定dtype的numpy数组再写入字节
                fout.write(np.array([tok], dtype=dtype).tobytes())
                pbar.update(1)


def create_memmap_from_binary(path: str, dtype=np.int32, dataset_name: str = "TinyStories"):

    binary_path = path.replace('.txt', '.bin')
    
    if not os.path.exists(binary_path):
        base_dir = os.path.join(os.path.dirname(path), dataset_name)
        vocab_path = os.path.join(base_dir, "vocab_str.json")
        merges_path = os.path.join(base_dir, "merges_str.json")
        special_tokens = ["<|endoftext|>"]
        tokenize_text_to_binary(path, binary_path, vocab_path, merges_path, special_tokens)
    # 创建memory map
    
    data = np.memmap(binary_path, dtype=dtype, mode='r')
    print(f"加载数据集: {binary_path}, 形状: {data.shape}, 数据类型: {data.dtype}")
    return data

def evaluate_model(model: nn.Module, eval_data: np.ndarray, batch_size: int, 
                  context_length: int, device: str, num_eval_batches: int = 10) -> float:
    """评估模型在验证集上的性能"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            try:
                inputs, targets = data_loading(eval_data, batch_size, context_length, device)
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                total_loss += loss.item()
                num_batches += 1
            except ValueError:
                # 如果数据不足，跳过
                break
    
    model.train()
    return total_loss / max(num_batches, 1)

def log_metrics(iteration: int, train_loss: float, eval_loss: float = None, 
               lr: float = None, elapsed_time: float = None, start_time: float = None,
               grad_norm: float = None,
               log_file: str = "training.log", experiment_name: str = "default"):
    """记录训练指标到多种格式，支持实验跟踪"""
    # 计算壁钟时间（从训练开始的总时间）
    wallclock_time = time.time() - start_time if start_time is not None else None
    
    # 构建日志字符串
    log_str = f"步数 {iteration:6d} | 训练损失: {train_loss:.4f}"
    if eval_loss is not None:
        log_str += f" | 验证损失: {eval_loss:.4f}"
    if lr is not None:
        log_str += f" | 学习率: {lr:.2e}"
    if elapsed_time is not None:
        log_str += f" | 步骤时间: {elapsed_time:.2f}s"
    if wallclock_time is not None:
        log_str += f" | 总时间: {wallclock_time:.2f}s"
    if grad_norm is not None:
        log_str += f" | 梯度范数: {grad_norm:.4f}"
    
    print(log_str)
    
    # 保存到文本日志文件
    with open(log_file, "a") as f:
        f.write(log_str + "\n")

    # 记录到 wandb
    if wandb is not None and getattr(wandb, "run", None) is not None:
        log_payload = {
            "step": iteration,
            "train/loss": train_loss,
        }
        if eval_loss is not None:
            log_payload["eval/loss"] = eval_loss
        if lr is not None:
            log_payload["train/lr"] = lr
        if elapsed_time is not None:
            log_payload["train/step_time_s"] = elapsed_time
        if wallclock_time is not None:
            log_payload["train/wallclock_s"] = wallclock_time
        if grad_norm is not None:
            log_payload["train/global_norm"] = grad_norm
        wandb.log(log_payload, step=iteration)

def training_together(args):
    """完整的训练函数"""
    
    # 设置设备和数据类型
    device = torch.device(args.device)
    torch_dtype = getattr(torch, args.dtype)
    np_dtype = getattr(np, args.data_dtype)
    
    print(f"使用设备: {device}")
    print(f"数据类型: {torch_dtype}")
    
    # 创建检查点目录
    os.makedirs(os.path.join(args.checkpoint_dir, args.experiment_name), exist_ok=True)
    
    # 加载数据集
    print("加载训练数据...")
    train_data = create_memmap_from_binary(args.train_data_path, dataset_name=args.dataset_name, dtype=np.int32)
    
    eval_data = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        print("加载验证数据...")
        eval_data = create_memmap_from_binary(args.eval_data_path, dataset_name=args.dataset_name, dtype=np.int32)
    
    # 初始化模型
    print("初始化模型...")
    model = Transformer_LM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch_dtype
    )
    model.to(device)
    
    
    # 计算模型参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {num_params:,}")
    
    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 恢复检查点（如果指定）
    start_iteration = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"从检查点恢复: {args.resume_from}")
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)
        print(f"从第 {start_iteration} 步继续训练")
    
    # 训练循环
    model.train()
    start_time = time.time()
    
    for iteration in range(start_iteration, args.max_iterations):
        # 学习率调度
        lr = learning_rate_schedule(
            iteration, args.max_learning_rate, args.min_learning_rate,
            args.warmup_iters, args.cosine_annealing_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        inputs, targets = data_loading(train_data, args.batch_size, args.context_length, str(device))
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        # 反向传播  
        optimizer.zero_grad()
        loss.backward()

        # 计算全局梯度范数（用于监控）
        def _compute_global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
            total_sq_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_sq_norm += float(param_norm.item() ** 2)
            return float(total_sq_norm ** 0.5)

        grad_norm = _compute_global_grad_norm(model.parameters())
        
        # 梯度裁剪
        gradient_clipping(model.parameters(), args.max_grad_norm)
        
        # 优化器步骤
        optimizer.step()
        
        # 定期记录训练损失
        if iteration % args.log_every == 0:
            elapsed_time = time.time() - start_time
            log_metrics(iteration, loss.item(), lr=lr, elapsed_time=elapsed_time, start_time=start_time, grad_norm=grad_norm, log_file=args.log_file)
        
        # 定期评估
        if eval_data is not None and iteration % args.eval_every == 0 and iteration > 0:
            eval_loss = evaluate_model(model, eval_data, args.batch_size, args.context_length, str(device))
            log_metrics(iteration, loss.item(), eval_loss=eval_loss, lr=lr, start_time=start_time, log_file=args.log_file)
        
        # 定期保存检查点
        if iteration % args.save_every == 0 and iteration > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, args.experiment_name, f"checkpoint_{iteration:06d}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
    
    # 保存最终检查点
    final_checkpoint_path = os.path.join(args.checkpoint_dir, args.experiment_name, "checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iterations, final_checkpoint_path)
    print(f"最终检查点已保存: {final_checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"训练完成！总用时: {total_time:.2f} 秒")

def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description="训练Transformer语言模型")
    
    # 数据参数
    parser.add_argument("--train_data_path", type=str, help="训练数据路径")
    parser.add_argument("--eval_data_path", type=str, help="验证数据路径")
    
    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=10000, help="词汇表大小")
    parser.add_argument("--context_length", type=int, default=256, help="上下文长度")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈网络维度")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta参数")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--max_iterations", type=int, default=10000, help="最大训练步数")
    parser.add_argument("--max_learning_rate", type=float, default=3e-4, help="最大学习率")
    parser.add_argument("--min_learning_rate", type=float, default=3e-5, help="最小学习率")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="学习率预热步数")
    parser.add_argument("--cosine_annealing_iters", type=int, default=8000, help="余弦退火步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--dataset_name", type=str, default="TinyStories", help="数据集名称")
    # 检查点和日志
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="检查点保存目录")
    parser.add_argument("--resume_from", type=str, help="从检查点恢复训练")
    parser.add_argument("--save_every", type=int, default=1000, help="检查点保存间隔")
    parser.add_argument("--eval_every", type=int, default=500, help="评估间隔")
    parser.add_argument("--log_every", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--experiment_name", type=str, default="default", help="实验名称")
    parser.add_argument("--log_file", type=str, default="training.log", help="日志文件")
    # 设备设置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="模型数据类型")
    parser.add_argument("--compile_model", action="store_true", help="编译模型以提高性能")
    parser.add_argument("--data_dtype", type=str, default="uint16", choices=["uint16", "int32", "int64"], help="数据文件数据类型")
    
    # 配置文件支持
    parser.add_argument("--config", type=str, help="从JSON配置文件加载参数")
    
    args = parser.parse_args()
    
    # 如果指定了配置文件，从中加载参数
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        # 用配置文件中的值更新args
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    # 保存配置到训练目录
    os.makedirs(os.path.join(args.checkpoint_dir, args.experiment_name), exist_ok=True)
    config_path = os.path.join(args.checkpoint_dir, args.experiment_name, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"训练配置已保存到: {config_path}")
    
    args.log_file = os.path.join(args.checkpoint_dir, args.experiment_name, "training.log")
    # 初始化 wandb（若可用），并记录实验配置
    if wandb is not None:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "cs336-basics"),
            name=args.experiment_name,
            config=vars(args),
        )
        # 声明 step 与 wallclock 作为横轴（可选，若版本支持）
        try:
            wandb.define_metric("step")
            wandb.define_metric("train/wallclock_s", step_metric="step")
        except Exception:
            pass


    training_together(args)

if __name__ == "__main__":
    main()
    # tokenize_text_to_binary("/data/yuer/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "/data/yuer/assignment1-basics/data/TinyStoriesV2-GPT4-valid.bin", vocab_path="/data/yuer/assignment1-basics/data/TinyStories_train/vocab_str.json", merges_path="/data/yuer/assignment1-basics/data/TinyStories_train/merges_str.json", special_tokens=["<|endoftext|>"])