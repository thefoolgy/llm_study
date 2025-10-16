import torch
from torch import nn
from torch.nn import init
from einops import rearrange, einsum
from jaxtyping import Float, Bool, Int


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device, dtype):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))
        std = (2 / (in_features + out_features))**0.5
        init.trunc_normal_(self.weight, mean = 0.0, std = std, a = -3*std, b = 3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        std = 1 
        init.trunc_normal_(self.weight, mean = 0.0, std = std, a = -3*std, b = 3*std)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model 
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.g = nn.Parameter(torch.ones(d_model, **factory_kwargs)) #initialize with one
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype 
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x * self.g) / rms
        return result.to(in_dtype)
    
def sigmoid(x: torch.Tensor): return 1 / (1 + torch.exp(-x))

def silu(x: torch.Tensor): return x * torch.sigmoid(x)

def glu(a: torch.Tensor, b: torch.Tensor): return a * b 

def swiglu(a: torch.Tensor, b: torch.Tensor): return glu(silu(a), b)

def compatible_dff(d_model: int) -> int:
    raw = (d_model * 8) / 3
    rounded = int((raw + 32) // 64) * 64 
    return rounded
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device, dtype):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff 
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.w1_weight = Linear(d_model, d_ff, **factory_kwargs)
        self.w2_weight = Linear(d_ff, d_model, **factory_kwargs)
        self.w3_weight = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x : torch.Tensor)-> torch.Tensor:
        w1_x = self.w1_weight(x)
        w3_x = self.w3_weight(x)
        h = swiglu(w1_x, w3_x)
        return self.w2_weight(h)
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.d_k = d_k 
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float()/d_k))
        positions = torch.arange(max_seq_len, device = device).float()
        freqs = torch.outer(positions, freq)

        self.register_buffer('cos_cached', torch.cos(freqs), persistent=False)
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.d_k:
            raise ValueError("x dim is not equal to d_k")
        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_even * sin_pos + x_odd * cos_pos
        out = torch.empty_like(x)
        out[..., 1::2] = out_odd
        out[..., ::2] = out_even
        return out
    
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = 1.0 / torch.sqrt(torch.tensor(d_k))

    def forward(self, 
                query: Float[torch.Tensor, "... seq_len_q d_k"],
                key: Float[torch.Tensor, "... seq_len_k d_k"],
                value: Float[torch.Tensor, "... seq_len_k d_v"],
                mask: Bool[torch.Tensor, "seq_len_q seq_len_k"] = None
    ) -> Float[torch.Tensor, "... seq_len_q d_v"]:
        attn_score = einsum(query, key, "... q d, ... k d -> ... q k") * self.scale
        if mask is not None:
            attn_score = attn_score.masked_fill(~mask, float("-inf"))
        attn_probs = softmax(attn_score, dim = -1)
        output = einsum(attn_probs, value, "... q k, ... k d -> ... q d")
        return output
    
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model : int, num_heads : int, max_seq_len: int, rope_theta: float = 10000.0, use_rope : bool = True,
                 device = None, dtype = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads 
        self.d_v = self.d_k
        self.use_rope = use_rope 

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [Linear(d_model, d_model, **factory_kwargs)
                                                              for _ in range (4)]
        self.attn = ScaledDotProductAttention(self.d_k)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype = torch.bool, device = device))
        self.register_buffer("casual_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(rope_theta, self.d_k, max_seq_len, device)
    
    def forward(self, 
                x: Float[torch.Tensor, "batch seq_len d_model"],
                token_positions: Int[torch.Tensor, "batch seq_len"] | None = None,
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        B, S, _ = x.shape
        q, k , v = [rearrange(proj(x), "b s (h d) -> b h s d", h = self.num_heads)
                    for proj in [self.q_proj, self.k_proj, self.v_proj]]
        if self.use_rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        out = self.attn(q, k, v, mask = self.casual_mask[..., :S, :S])
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, rope_theta: float = 10_000.0,
                 use_rope: bool = True, device = None, dtype = None) -> None:
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.norm1 = RMSNorm(d_model, **kwargs)
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len,
            rope_theta,
            use_rope,
            **kwargs,
        )
        self.norm2 = RMSNorm(d_model, **kwargs)
        self.ff = SwiGLU(d_model, d_ff, **kwargs)
    
    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor | None = None) -> torch.Tensor:
        b, s, _ = x.shape
        attn_out = self.attn(self.norm1(x), token_positions)
        x = x + attn_out 
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out 
        return x

def _copy_param(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.shape == source.shape:
        target.data.copy_(source)
    elif source.T.shape == target.shape:
        target.data.copy(source.T)
    else:
        raise ValueError("target and source are not in same shape")
    
class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size : int,
                 context_length: int, 
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device=None,
                 dtype=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype)
        self.token_emb = Embedding(vocab_size, d_model, **kw)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                use_rope=True,
                **kw,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, **kw)
        self.lm_head = Linear(d_model, vocab_size, **kw)
        self.context_length = context_length
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        if s > self.context_length:
            raise ValueError("sequence length exceed the max seq len")
        x = self.token_emb(token_ids)
        pos = torch.arange(s, device = token_ids.device).unsqueeze(0).expand(b,s)
        for blk in self.blocks:
            x = blk(x, token_positions = pos)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

        

