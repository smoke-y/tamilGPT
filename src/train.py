import os
import math
import torch
from model import *
from transformers import GPT2Tokenizer

EPOCH = 10
TOTAL_BATCH_SIZE = 524288
B = 2
T = 1024

off = B*T
gradeAccumulationSteps = TOTAL_BATCH_SIZE // (T*B)
tokenizer = GPT2Tokenizer.from_pretrained("Lagstill/GPT-2-Tamil")
if not os.path.exists("misc/"): os.makedirs("misc/")

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
        if self.tokens.nelement() - self.cursor > off+1:
            chunk = self.tokens[self.cursor:self.cursor+off+1]
            self.cursor += off
            return [chunk[:-1].view(B, T), chunk[1:].view(B, T)]
        else:
            self.fileId = (self.fileId + 1) % self.filesCount
            self.loadCurFileId()
            return self.nextChunk()

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
def optimizerHook(param):
    optimizer.step()
    optimizer.zero_grad()

deviceType = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(deviceType)
torch.set_float32_matmul_precision("high")
print("Using", device)
#NOTE - vocab: 50000, but our gpt has 50257 embeddings
model = GPT.from_pretrained("gpt2")
model.applyLoRa()
model.freezeNonLoRa()
if os.path.exists("misc/weights.lora"): model.loadLoRaWeights("misc/weights.lora")
model.to(device)
model = torch.compile(model)
optimizer = model.createOptimizer(weightDecay=0.1, lr=6e-4, device=device)
for p in model.parameters(): p.register_post_accumulate_grad_hook(optimizerHook)
optimizer.zero_grad()

chunugs = Chungus()
x = torch.empty((B, T), dtype=torch.long, device=device)
y = torch.empty((B, T), dtype=torch.long, device=device)
off = B*T
step = 0
log = open("misc/trace.log", "w+")
try:
    for epoch in range(EPOCH):
        for step in range(max_steps):
            if step <= max_steps+1:
                lr = get_lr(step)
                for param_group in optimizer.param_groups: param_group["lr"] = lr
                step += 1
            lossAcum = 0.0
            for microStep in range(gradeAccumulationSteps):
                xChunk,yChunk = chunugs.nextChunk()
                x.copy_(xChunk)
                y.copy_(yChunk)
                with torch.autocast(device_type=deviceType, dtype=torch.bfloat16):
                    pred, loss = model.forward(x, y)
                loss /= gradeAccumulationSteps
                lossAcum += loss.detach().cpu().numpy()
                loss.backward()
            log.write(f"{lossAcum}\n")
except Exception as e: print(e.with_traceback())
finally:
    model.saveLoRaWeights("misc/weights.lora")
    log.close()