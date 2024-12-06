import math
import torch
from model import *
from datasets import load_dataset
from transformers import GPT2Tokenizer

class Chungus:
    def __init__(self, B: int, T: int, tokenizer) -> None:
        self.B, self.T= B, T
        self.tokenizer = tokenizer
    def chunk(self, toks):
        length = len(toks) - 1
        chunks = []
        cursor = 0
        off = self.B*self.T
        while length > 1:
            chunk = toks[cursor:cursor+off+1]
            x = torch.tensor(chunk, dtype=torch.long)[:-1].view(self.B, self.T)
            y = torch.tensor(chunk, dtype=torch.long)[1:].view(self.B, self.T)
            chunks.append([x,y])
            cursor += off
            length -= off
            if length < off: break
        return chunks
    
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    #SECTION - https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L349
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

deviceType = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", deviceType)
#NOTE - vocab: 50000, but our gpt has 50257 embeddings
tokenizer = GPT2Tokenizer.from_pretrained("Lagstill/GPT-2-Tamil")
ds = load_dataset("uonlp/CulturaX", "ta", split="train", streaming=True)
dc = Chungus(2, 6, None)
model = GPT.from_pretrained("gpt2")
model.applyLoRa()
model.freezeNonLoRa()
optimizer = model.createOptimizer(weightDecay=0.1, lr=6e-4, device="cpu")
torch.autograd.set_detect_anomaly(True)
for data in ds:
    d = tokenizer.encode(data["text"])
    chunks = dc.chunk(d)
    for chunk in chunks:
        pred, loss = model.forward(chunk[0], chunk[1])
        optimizer.zero_grad()
        print(loss)
        loss.backward()
        optimizer.step()