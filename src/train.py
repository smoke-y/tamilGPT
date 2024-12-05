from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("Hemanth-thunder/Tamil-Mistral-7B-v0.1",add_prefix_space=True)
ds = load_dataset("uonlp/CulturaX", "ta", split="train", streaming=True)