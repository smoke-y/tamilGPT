import os
import torch
from model import *
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("Lagstill/GPT-2-Tamil")
if not os.path.exists("misc/"): os.makedirs("misc/")
hyp = Hyperparameters()

class Chungus:
    def __init__(self) -> None:
        self.filesCount = len(os.listdir("data/"))
        self.fileId = 0
        self.tokens = None
        self.cursor = 0
        self.loadCurFileId()
    def loadCurFileId(self) -> None:
        file = open(f"data/{self.fileId}.txt", "r", encoding="utf8")
        self.tokens = torch.tensor(tokenizer.encode(file.read()), device="cpu", dtype=torch.long)
        file.close()
        self.cursor = 0
    def nextChunk(self):
        off = hyp.batch*hyp.seq_len
        if self.tokens.nelement() - self.cursor > off+1:
            chunk = self.tokens[self.cursor:self.cursor+off+1]
            self.cursor += off
            return chunk[:-1].view(hyp.batch, hyp.seq_len), chunk[1:].view(hyp.batch, hyp.seq_len)
        else:
            self.fileId = (self.fileId + 1) % self.filesCount
            self.loadCurFileId()
            return self.nextChunk()

deviceType = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(deviceType)
torch.set_float32_matmul_precision("high")
print("Using", device)
model = GPT(Config())
model.to(device)
optimizers, schedulers = model.create_optimizers(hyp)
model = torch.compile(model)

chungus = Chungus()
x = torch.empty((hyp.batch, hyp.seq_len), dtype=torch.long, device=device)
y = torch.empty((hyp.batch, hyp.seq_len), dtype=torch.long, device=device)
log = open("misc/trace.log", "w+")
optimizer2 = optimizers[1]
for step in range(hyp.num_iterations+1):
    xChunk, yChunk = chungus.nextChunk()
    x.copy_(xChunk)
    y.copy_(yChunk)
    with torch.autocast(device_type=deviceType, dtype=torch.bfloat16):
        pred, loss = model.forward(x, y)
    loss.backward()
    loss = loss.detach().cpu().numpy()
    # momentum warmup for Muon
    frac = min(step / 300, 1)
    for group in optimizer2.param_groups:
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
        opt.zero_grad()
    log.write(f"{loss}\n")