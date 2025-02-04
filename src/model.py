import torch
import torch.nn as nn
from muon import Muon
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    nheads:  int = 12
    embdim:  int = 768
    layers:  int = 8
    maxseq:  int = 1024
    vocab:   int = 50257
@dataclass
class Hyperparameters:
    batch = 1
    seq_len = 1024
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # optimization
    num_iterations = 1770 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int): super().__init__(in_features, out_features, bias=False)
    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad(): self.weight.uniform_(-bound, bound)
    def forward(self, x: torch.Tensor): return F.linear(x, self.weight.type_as(x))

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

def norm(x: torch.Tensor) -> torch.Tensor: return nn.functional.rms_norm(x, (x.size(-1),))
def next_multiple_of_n(v: float | int, *, n: int): return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class CasualSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        dim = config.embdim
        assert dim % config.nheads == 0
        #q,k,v weights
        self.c_attn = CastedLinear(dim, 3*dim)
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.detach().zero_()
        self.rotary = Rotary(dim // config.nheads)
        self.nheads = config.nheads
        self.attn_scale = 0.12
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size() #Batch, Seq, Emb-dim
        qkv = self.c_attn(x)
        q,k,v = qkv.split(C, dim=2)
        q = q.view(B, T, self.nheads, C // self.nheads)
        k = k.view(B, T, self.nheads, C // self.nheads)
        v = v.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        q,k = norm(q), norm(k)
        q,k = self.rotary(q).transpose(1,2), self.rotary(k).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True, scale=self.attn_scale)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y)
class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        dim = config.embdim
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()
    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        return self.c_proj(F.relu(x).square())
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
        self.word_embedding = nn.Embedding(next_multiple_of_n(config.vocab, n=128), config.embdim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.ln_f = nn.LayerNorm(config.embdim)
        self.lm_head = CastedLinear(config.embdim, next_multiple_of_n(config.vocab, n=128))
        self.lm_head.weight.detach().zero_()
        self.num_encoding_layers = config.layers // 2
        self.num_decoding_layers = config.layers - self.num_encoding_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoding_layers))
    def forward(self, x: torch.Tensor, groundTruth = None):
        B, T = x.size()
        assert T <= self.config.maxseq, "sequence length > max sequence length"
        skip_connections = []
        x = norm(self.word_embedding(x))
        for i in range(self.num_encoding_layers):
            x = self.blocks[i](x)
            skip_connections.append(x)
        for i in range(self.num_decoding_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoding_layers+i](x)
        x = self.ln_f(norm(x))
        logits = self.lm_head(x)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = None
        if groundTruth is not None: loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), groundTruth.view(-1))
        return logits, loss
    def create_optimizers(self, hyp: Hyperparameters) -> list:
        hidden_matrix_params = [p for n, p in self.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
        embed_params = [p for n, p in self.named_parameters() if "embed" in n]
        scalar_params = [p for p in self.parameters() if p.ndim < 2]
        head_params = [self.lm_head.weight]
        adam_params = [dict(params=head_params, lr=0.008), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
        optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95)
        optimizers = [optimizer1, optimizer2]

        def get_lr(step: int):
            t = 1 - step / hyp.num_iterations # time remaining in training
            assert 1 >= t >= 0
            w = min(t / hyp.cooldown_frac, 1.0) # 1 -> 0
            return w * 1.0 + (1 - w) * 0.1
        schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
        return optimizers, schedulers
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