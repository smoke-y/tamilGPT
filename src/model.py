import torch
import torch.nn as nn

class Config:
    nheads: int = 12
    embdim: int = 24

class CasualSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.embdim % config.nheads == 0
        self.nheads = config.nheads
        #q,k,v weights
        self.attn = nn.Linear(config.embdim, 3*config.embdim)
        self.proj = nn.Linear(config.embdim, config.embdim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size() #Batch, Seq, Emb-dim
        qkv = self.attn(x)
        q,k,v = qkv.split(C, dim=2)
        q = q.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        k = k.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        v = v.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        y = nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.proj(y)
class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.l1 = nn.Linear(config.embdim, config.embdim * 4)
        self.act = nn.GELU()
        self.l2 = nn.Linear(config.embdim * 4, config.embdim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.act(x)
        return self.l2(x)
class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embdim)
        self.attn = CasualSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embdim)
        self.mlp = MLP(config)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))