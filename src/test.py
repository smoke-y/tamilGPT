import torch
from model import *
from transformers import GPT2Tokenizer

prompts = [
    "நான் ஒரு பெரிய மொழி மாதிரி, ",
    "எனக்கு இனிப்பு பிடிக்கும் மற்றும், "
]

tokenizer = GPT2Tokenizer.from_pretrained("Lagstill/GPT-2-Tamil")
deviceType = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(deviceType)
torch.set_float32_matmul_precision("high")
print("Using", device)

model = GPT(Config())
state_dict = torch.load("misc/weights.pth", weights_only=True)
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()})
model.to(device)
model = torch.compile(model)

with torch.no_grad():
    for prompt in prompts:
        print("PROMPT:", prompt)
        inp = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
        inp = model.generate(inp.unsqueeze(0), 50)
        print("RESPONSE:", tokenizer.decode(inp.detach().squeeze(0).cpu()), end="\n\n")