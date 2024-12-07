import math
import torch
from model import *
from os import listdir
from transformers import GPT2Tokenizer

EPOCH = 10
B = 2
T = 24

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

files = ["data/"+f for f in listdir("data/")]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)
#NOTE - vocab: 50000, but our gpt has 50257 embeddings
tokenizer = GPT2Tokenizer.from_pretrained("Lagstill/GPT-2-Tamil")
model = GPT.from_pretrained("gpt2")
model.applyLoRa()
model.freezeNonLoRa()
model.to(device)
optimizer = model.createOptimizer(weightDecay=0.1, lr=6e-4, device=device)

x = torch.empty((B, T), dtype=torch.long, device=device)
y = torch.empty((B, T), dtype=torch.long, device=device)
off = B*T
for epoch in range(EPOCH):
    for file in files:
        file = open(file, "r", encoding="utf8")
        tokens = torch.tensor(tokenizer.encode(file.read()), device="cpu", dtype=torch.long)
        file.close()
        length = tokens.nelement() - 1
        cursor = 0
        while length > off:
            chunk = tokens[cursor:cursor+off+1]
            x.copy_(chunk[:-1].view(B, T))
            y.copy_(chunk[1:].view(B, T))
            cursor += off
            length -= off

            pred, loss = model.forward(x, y)
            optimizer.zero_grad()
            print(loss)
            loss.backward()
            optimizer.step()