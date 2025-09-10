import numpy as np
import torch

def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将单一连接的 token 序列 x (np.int_ / np.uint* / np.int64 等整型) 
    采样为一个批次的 (inputs, targets)，形状均为 (B, m)，放置到给定 device。
    
    inputs[b]  = x[s_b : s_b + m]
    targets[b] = x[s_b + 1 : s_b + m + 1]
    其中 s_b 在 [0, n - m - 2] 闭区间内均匀采样。
    """
    n = x.shape[0]
    m = context_length
    if n < m + 1:
        raise ValueError(f"序列过短：需要至少 m+1={m+1} 个 token，当前 n={n}")
    # 随机数源
    rng = np.random.default_rng()
    # 起点范围：确保右移一位后的 targets 仍有 m 个元素
    # 允许的最大起点是 n - (m+1)
    max_start = n - (m + 1)
    starts = rng.integers(low=0, high=max_start + 1, size=(batch_size,), dtype=np.int64)  # [B]
    # 通过广播构造索引矩阵 [B, m]
    offs = np.arange(m, dtype=np.int64)  # [m]
    idx_in  = starts[:, None] + offs            # inputs 索引
    idx_tgt = starts[:, None] + offs + 1        # targets 索引（右移一位）
    batch_x_np   = x[idx_in]
    batch_y_np   = x[idx_tgt]
    # 转为 torch.long 并放到指定 device
    batch_x = torch.as_tensor(batch_x_np, dtype=torch.long, device=device)
    batch_y = torch.as_tensor(batch_y_np, dtype=torch.long, device=device)
    return batch_x, batch_y