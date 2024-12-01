from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Hemanth-thunder/Tamil-Mistral-7B-v0.1",add_prefix_space=True)