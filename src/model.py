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

# Define the parameters
batch_size = 2
sequence_length_1 = 12
sequence_length_2 = 24
# Calculate the total number of elements needed
total_elements = batch_size * sequence_length_1 * sequence_length_2
# Generate a sequence of numbers from 1 to total_elements
numbers = torch.arange(1, total_elements + 1, dtype=torch.float)
# Reshape the tensor into the desired shape: (batch_size, sequence_length_1, sequence_length_2)
numbers = numbers.view(batch_size, sequence_length_1, sequence_length_2)
# Print the generated sequences
print(numbers.size())

config = Config()
csa = CasualSelfAttention(config)
csa.forward(numbers)