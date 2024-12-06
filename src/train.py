from transformers import GPT2Tokenizer
from datasets import load_dataset
import torch

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
            x = torch.tensor(chunk, dtype=torch.int)[:-1].view(self.B, self.T)
            y = torch.tensor(chunk, dtype=torch.int)[1:].view(self.B, self.T)
            chunks.append([x,y])
            cursor += off
            length -= off
            if length < off: break
        return chunks

#NOTE - vocab: 50000, but our gpt has 50257 embeddings
tokenizer = GPT2Tokenizer.from_pretrained("Lagstill/GPT-2-Tamil")
print(tokenizer)
ds = load_dataset("uonlp/CulturaX", "ta", split="train", streaming=True)
dc = Chungus(2, 6, None)
for data in ds:
    d = tokenizer.encode(data["text"])
    chunks = dc.chunk(d)