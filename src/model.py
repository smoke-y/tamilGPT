import torch
import inspect
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    nheads:  int = 12
    embdim:  int = 768
    layers:  int = 12
    maxseq:  int = 1024
    vocab:   int = 50257

class CasualSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.embdim % config.nheads == 0
        #q,k,v weights
        self.c_attn = nn.Linear(config.embdim, 3*config.embdim)
        self.c_proj = nn.Linear(config.embdim, config.embdim)
        self.nheads = config.nheads
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size() #Batch, Seq, Emb-dim
        qkv = self.c_attn(x)
        q,k,v = qkv.split(C, dim=2)
        q = q.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        k = k.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
        v = v.view(B, T, self.nheads, C // self.nheads).transpose(1,2)
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
class LoRa(nn.Module):
    def __init__(self, featureIn: int, featureOut: int, rank: int, alpha: float) -> None:
        super().__init__() 
        self.lora_a = nn.Parameter(torch.zeros(rank, featureOut))
        self.lora_b = nn.Parameter(torch.zeros(featureIn, rank))
        self.scale = alpha / rank
    def forward(self, originalWeight: torch.Tensor) -> torch.Tensor: return originalWeight + (self.lora_b @ self.lora_a)*self.scale
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
    @classmethod
    def from_pretrained(cls, model_type):
        #SECTION - https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L131
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        # layers, nheads and embdim are determined from model_type
        config_args = {
            'gpt2':         dict(layers=12, nheads=12, embdim=768),  # 124M params
            'gpt2-medium':  dict(layers=24, nheads=16, embdim=1024), # 350M params
            'gpt2-large':   dict(layers=36, nheads=20, embdim=1280), # 774M params
            'gpt2-xl':      dict(layers=48, nheads=25, embdim=1600), # 1558M params
        }[model_type]
        config_args['vocab'] = 50257 # always 50257 for GPT model checkpoints
        config_args['maxseq'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        #for i in sd_keys_hf: print(i)
        #for i in sd_keys: print(i)
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        return model
    def applyLoRa(self, rank: int = 4, alpha: float = 32.0):
        def applyLoraToLayer(layer):
            nonlocal NonLoRaParam, LoRaParam
            if hasattr(layer, "weight"):
                layer.requires_grad = False
                NonLoRaParam += layer.weight.nelement()
                if hasattr(layer, "bias"):
                    if layer.bias is not None: NonLoRaParam += layer.bias.nelement()
                if len(layer.weight.shape) == 1: return
                torch.nn.utils.parametrize.register_parametrization(
                    layer, "weight", LoRa(*layer.weight.shape, rank, alpha)
                )
                loraParam = layer.parametrizations["weight"][0]
                LoRaParam += loraParam.lora_a.nelement() + loraParam.lora_b.nelement()

        def recurseAndApply(module):
            for name, child in module.named_children(): recurseAndApply(child)
            if hasattr(module, "weight"): applyLoraToLayer(module)

        with torch.no_grad():
            NonLoRaParam = 0
            LoRaParam = 0
            recurseAndApply(self)
            print(
                f"Original param count: {NonLoRaParam}\n"
                f"LoRa: {LoRaParam}\n"
                f"Total: {LoRaParam + NonLoRaParam}\n"
                f"Increment: {(LoRaParam / NonLoRaParam) * 100:.2f}%"
            )
    def createOptimizer(self, weightDecay: float, lr: float, device: str):
        decayParams = []
        nonDecayParams = []
        for name, param in self.named_parameters():
            if param.requires_grad == False: continue
            if param.dim() >= 2: decayParams.append(param)
            else: nonDecayParams.append(param)
        optimGroups = [
            {"params": decayParams, "weight_decay": weightDecay},
            {"params": nonDecayParams, "weight_decay": 0.0},
        ]
        fusedAdam = ("fused" in inspect.signature(torch.optim.AdamW).parameters) and (device == "cuda")
        if fusedAdam: print("Using fused AdamW")
        return torch.optim.AdamW(optimGroups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fusedAdam)
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
    def saveLoRaWeights(self, fileName: str):
        loraWeights = {}
        for name, layer in self.named_parameters():
            if name.endswith("lora_a") or name.endswith("lora_b"): loraWeights[name] = layer.detach().cpu()
        torch.save(loraWeights, fileName)
        print("LoRa weights saved to", fileName)
    def loadLoRaWeights(self, fileName: str):
        loraWeights = torch.load(fileName, weights_only=True)
        for name, layer in self.named_parameters():
            if name.endswith("lora_a") or name.endswith("lora_b"): layer.data.copy_(loraWeights[name])
        print("LoRa weights loaded from", fileName)