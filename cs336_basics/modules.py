import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device : torch.device|None =None, dtype : torch.dtype| None =None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 使用高斯分布初始化权重：均值=0，方差=2/(d_in+d_out)，截断到3标准差内
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
         

class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device|None=None, dtype: torch.dtype|None=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # (bs, seq, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # 计算每个元素的平方
        x_squared = x * x
        # 计算平方和并除以d_model
        mean_squared = torch.mean(x_squared, dim=-1, keepdim=True)
        # 开根号并加上eps防止除零
        rms = torch.sqrt(mean_squared + self.eps)
        # RMS归一化
        result = x / rms * self.weight
        return result.to(in_dtype)
        

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        half = d_k // 2
        k = torch.arange(half, device=device).float() # latent dimension
        i = torch.arange(max_seq_len, device=device).float()
        freq = theta ** (-2 * k / d_k) # (half)
        freq_i =  torch.einsum('i,j->ij', i, freq) # (max_seq_len, half)
        
        self.register_buffer('cos', torch.cos(freq_i), persistent=False)
        self.register_buffer('sin', torch.sin(freq_i), persistent=False)
        
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        # 旋转时 奇数维度为负 cos(theta) * x_0  + sin(theta) * x_1, 偶数维度为正 sin(theta) * x_0  + cos(theta) * x_1
        '''Rotate half the hidden dims of the input.'''
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:   
        '''Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note that you should tolerate x with an arbitrary number of batch dimensions. You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.'''
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        cos = torch.stack((cos, cos), dim=-1).reshape(*cos.shape[:-1], self.d_k).to(self.device)
        sin = torch.stack((sin, sin), dim=-1).reshape(*sin.shape[:-1], self.d_k).to(self.device)
        x = x * cos + self.rotate_half(x) * sin
        return x
        


def softmax(x: torch.tensor, dim:int):
    assert dim < x.dim()
    x_sub = x.sub(x.max(dim=dim, keepdim=True)[0])
    x_exp = x_sub.exp()
    return x_exp / x_exp.sum(dim=dim, keepdim=True)



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device|None = None, dtype: torch.dtype|None=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff # d_ff = 8d_model/3
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
    def forward(self, x: torch.tensor):
        proj = self.w1(x)
        gate = proj * torch.sigmoid(proj)
        val = self.w3(x)
        h = gate * val
        return self.w2(h)
        

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device|None =None, dtype:torch.dtype|None =None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 使用torch.nn.init.trunc_normal_初始化权重
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
    
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor|None = None) -> torch.Tensor:
    d_k = q.shape[-1]
    attn = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype))
    if mask is not None:
        attn = attn.masked_fill(~mask, -torch.inf)
    val = softmax(attn, dim=-1) @ v
    return val

    
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope : RoPE|None = None, device: torch.device|None = None, dtype: torch.dtype|None = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.rope = rope
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 应该使用整数除法
        self.device = device
        self.dtype = dtype

        # 投影层
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor|None = None, token_positions: torch.Tensor|None = None) -> torch.Tensor:
        bs = x.shape[0]
        
        q = self.w_q(x).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2) # 希望按“每个 batch、每个 head”独立做注意力
        k = self.w_k(x).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)  
        v = self.w_v(x).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        if self.rope is not None:
            BH = self.num_heads * bs
            q_ = q.reshape(BH, -1, self.d_k)
            k_ = k.reshape(BH, -1, self.d_k)
            if token_positions is None:
                token_positions = torch.arange(q.shape[-2], device=self.device)
            q = self.rope(q_, token_positions).reshape(bs, self.num_heads, -1, self.d_k)
            k = self.rope(k_, token_positions).reshape(bs, self.num_heads, -1, self.d_k)
            
        causal_mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool)) # True allow Flase forbidden
        attn = scaled_dot_product_attention(q, k, v, causal_mask)
        attn = attn.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        out = self.w_o(attn)
        return out
    
    