import os
import torch
import traceback
from model import *
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("Lagstill/GPT-2-Tamil")
if not os.path.exists("misc/"): os.makedirs("misc/")
hyp = Hyperparameters()
deviceType = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(deviceType)
torch.set_float32_matmul_precision("high")
print("Using", device)

GEN_TENS = torch.tensor(tokenizer.encode("வணக்கம், நான்"), dtype=torch.long, device=device).unsqueeze(0)

class Chungus:
    def __init__(self) -> None:
        self.file = open("data/ta.txt", "r", encoding="utf8")
        self.tokens = None
        self.readChunk()
    def readChunk(self) -> None:
        read_len = hyp.chungus_file_stream_len * hyp.batch + 1
        chunk = self.file.read(read_len)
        if len(chunk) < read_len:
            self.file.seek(0)
            chunk = self.file.read(read_len)
        if type(self.tokens) != type(None):
            self.tokens = torch.cat([self.tokens, torch.tensor(tokenizer.encode(chunk), device="cpu", dtype=torch.long)])
        else: self.tokens = torch.tensor(tokenizer.encode(chunk), device="cpu", dtype=torch.long)
    def nextChunk(self):
        off = hyp.batch*hyp.seq_len + 1
        if self.tokens.nelement() < off: self.readChunk()
        chunk = self.tokens[:off]
        self.tokens = self.tokens[off:]
        return (chunk[:-1]).view(hyp.batch, hyp.seq_len), (chunk[1:]).view(hyp.batch, hyp.seq_len)
    def close(self) -> None: self.file.close()

model = GPT(Config())
model.to(device)
optimizers, schedulers = model.create_optimizers(hyp)
model = torch.compile(model)

def save_weights() -> None: torch.save(model.state_dict(), "misc/weights.pth")

chungus = Chungus()
x = torch.empty((hyp.batch, hyp.seq_len), dtype=torch.long, device=device)
y = torch.empty((hyp.batch, hyp.seq_len), dtype=torch.long, device=device)
log = open("misc/trace.log", "w+")
gen = open("misc/gen.log", "w+", encoding="utf8")
optimizer2 = optimizers[1]
step = 0
warmup = True
try:
    while True:
    ################# FORWARD #################
        xChunk, yChunk = chungus.nextChunk()
        x.copy_(xChunk)
        y.copy_(yChunk)
        with torch.autocast(device_type=deviceType, dtype=torch.bfloat16):
            pred, loss = model.forward(x, y)
    ################# BACKWARD #################
        loss.backward()
        loss = loss.detach().cpu().numpy()
        frac = min(step / 300, 1) if warmup else 1
        for group in optimizer2.param_groups:
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
            opt.zero_grad()
    ################# LOG/UPDATE/SAVE/GENERATE #################
        log.write(f"{loss}\n")
        print(f"\rcurrent_loss: {loss}", end="")
        step += 1
        if step >= 300:
            warmup = False
            step = 0
            with torch.no_grad():
                inp = model.generate(GEN_TENS, 10)
                gen.write("###\n" + str(tokenizer.decode(inp.detach().squeeze(0).cpu())) + "\n##\n")
        if step % 50 == 0: save_weights()
except: traceback.print_exc()
finally:
    save_weights()
    chungus.close()