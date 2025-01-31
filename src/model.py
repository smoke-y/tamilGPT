import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    nheads:  int = 12
    embdim:  int = 768
    layers:  int = 1
    maxseq:  int = 1024
    vocab:   int = 50257

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=65536):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)
    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CasualSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        dim = config.embdim
        assert dim % config.nheads == 0
        #q,k,v weights
        self.c_attn = nn.Linear(dim, 3*dim)
        self.c_proj = nn.Linear(dim, dim)
        self.rotary = Rotary(dim // config.nheads)
        self.nheads = config.nheads
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size() #Batch, Seq, Emb-dim
        qkv = self.c_attn(x)
        q,k,v = qkv.split(C, dim=2)
        q = q.view(B, T, self.nheads, C // self.nheads)
        k = k.view(B, T, self.nheads, C // self.nheads)
        v = v.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        q,k = self.rotary(q).transpose(1,2), self.rotary(k).transpose(1,2)
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y)
class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.embdim, config.embdim * 4)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.embdim * 4, config.embdim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        return self.c_proj(self.act(x))
class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embdim)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embdim)
        self.mlp = MLP(config)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))
class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab, config.embdim),
            wpe = nn.Embedding(config.maxseq, config.embdim),
            h = nn.ModuleList([Block(config) for _ in range(config.layers)]),
            ln_f = nn.LayerNorm(config.embdim),
        ))
        self.lm_head = nn.Linear(config.embdim, config.vocab, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   #first and last share params
    def forward(self, x: torch.Tensor, groundTruth = None):
        B, T = x.size()
        assert T <= self.config.maxseq, "sequence length > max sequence length"
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        posEmb = self.transformer.wpe(pos)
        tokEmb = self.transformer.wte(x)
        x = tokEmb + posEmb
        for block in self.transformer.h: x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if groundTruth is not None: loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), groundTruth.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.maxseq else idx[:, -self.config.maxseq:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx