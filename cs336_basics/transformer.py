import torch
import torch.nn as nn
from cs336_basics.modules import *
from cs336_basics.tokenizer import Tokenizer

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device: torch.device, dtype: torch.dtype, rope: RoPE|None = None):
        """
        d_model: int Dimensionality of the Transformer block inputs.  
        num_heads: int Number of heads to use in multi-head self-attention. 
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.device = device
        self.dtype = dtype
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.RMSnorm1 = RMSnorm(d_model, device=device, dtype=dtype)
        self.RMSnorm2 = RMSnorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.RMSnorm1(x)) + x
        h = self.ffn(self.RMSnorm2(h)) + h
        return h

class Transformer_LM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device, dtype: torch.dtype):
        """
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.  
        context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.  
        num_layers: int The number of Transformer blocks to use.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList([Transformer_block(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype) for _ in range(num_layers)])
        self.RMSnorm = RMSnorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.RMSnorm(x)
        return self.lm_head(x)

    def decode(self, tokenizer: Tokenizer, prompt: str = "", max_new_tokens: int = 100, temperature: float = 1.0, top_p: float = 0.95) -> str:
        # 将prompt编码为tensor
        x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            # 检查是否已经生成了结束token
            if x[0, -1].item() == tokenizer.vocab["<|endoftext|>"]:
                break
                
            # 前向传播获取logits
            logits = self.forward(x)
            logits = logits[:, -1, :] / temperature
            
            # 应用top-p采样
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 创建mask：保留累积概率 <= top_p的token
            mask = cumsum_probs <= top_p
            # 确保至少保留一个token
            mask[:, 0] = True
            
            # 将mask应用到原始概率分布
            sorted_probs = sorted_probs * mask
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            
            # 从排序后的概率中采样
            next_token_idx = torch.multinomial(sorted_probs, 1)
            # 将索引映射回原始token索引
            next_token = sorted_indices.gather(-1, next_token_idx)
            
            # 将新token添加到序列中
            x = torch.cat([x, next_token], dim=-1)
            
        # 将tensor转换回list并解码
        return tokenizer.decode(x[0].tolist())