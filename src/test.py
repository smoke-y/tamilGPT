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
model = GPT.from_pretrained("gpt2")
model.applyLoRa()
model.loadLoRaWeights("misc/weights.lora")
model.to(device)
model = torch.compile(model)

with torch.no_grad():
    sampleRNG = torch.Generator(device=device)
    for prompt in prompts:
        print("PROMPT:", prompt)
        inp = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
        inp = inp.unsqueeze(0)
        for i in range(10):
            logits,_ = model.forward(inp)
            logits = logits[:,-1,:]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topkProbs, topkIndices = torch.topk(probs, 50)
            ix = torch.multinomial(topkProbs, 1, generator=sampleRNG)
            xcol = torch.gather(topkIndices, -1, ix)
            inp = torch.cat((inp, xcol), dim=1)
        print("RESPONSE:", tokenizer.decode(inp.detach().squeeze(0).cpu()), end="\n\n")