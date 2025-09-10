from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    t = state.get("t", 0)
                    grad = p.grad.data
                    p.data -= lr / math.sqrt(t+1) * grad
                    state["t"] = t + 1
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1 parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2 parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is not None:
                    grad = p.grad.data
                    state = self.state[p]
                    t = state.get("t", 1)
                    if t == 1:
                        state["exp_avg"] = grad.clone().zero_()
                        state["exp_avg_sq"] = grad.clone().zero_()
                        
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) # m ← β1m + (1− β1)g
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v ← β2v + (1− β2)g^2
                    lr_t = lr * math.sqrt(1 - beta2**(t)) / (1 - beta1**(t)) # √(1− β2^t)/(1− β1^t)
                    with torch.no_grad():
                        p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr_t)
                        p.mul_(1 - lr * weight_decay)
                    state["t"] = t + 1
                    
                    


def learning_rate_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_annealing_iters: int) -> float:
    
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_annealing_iters:
        import math
        progress = (it - warmup_iters) / (cosine_annealing_iters - warmup_iters)
        cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_term
    else:
        return min_learning_rate
    
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 需要计算所有参数梯度的全局L2范数，而不是单独处理每个参数
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 如果全局范数超过最大值，则按比例缩放所有梯度
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)