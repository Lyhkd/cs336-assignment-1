import torch

def cross_entropy(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    print(logits.shape, targets.shape)
    logits = logits.view(-1, logits.shape[-1]) # bs, vocab_size or batch_size, seq_len, vocab_size -> (B*T, vocab_size)
    targets = targets.view(-1) # shape = batch_size, seq_len or batch_size -> (B*T,)
    shift_logits = logits.sub(logits.max(dim=-1, keepdim=True)[0]) # xi 减去最大
    log_exp_sum = torch.log(shift_logits.exp().sum(dim=-1, keepdim=True)) # 计算分母，并转换为log
    true_logits = shift_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # 找到正确的logit 它会从 input 的 dim 维度里按 index 提供的位置取元素。index 的形状必须和 input 除了 dim 以外的其他维度完全一致 ->  (B*T, )
    prob = log_exp_sum - true_logits
    return prob.mean()