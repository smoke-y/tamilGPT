import os
import re
import emoji
from datasets import load_dataset

FILE_SIZE = 0.01         #GB. Each data file is FILE_SIZE big
DATASET_SIZE = 11        #GB. Size of the entire dataset
GB_TO_BYTES = 1024 ** 3

def get_emoji_pattern():
    # Sort emojis by length to ensure multi-character emojis are matched first
    emojis = sorted(emoji.EMOJI_DATA.keys(), key=len, reverse=True)
    pattern = '|'.join(re.escape(em) for em in emojis)
    return re.compile(f'({pattern})')

os.makedirs("data/", exist_ok=True)
id = 0
file = open(f"data/{id}.txt", 'w', encoding="utf8")
size = 0

emojiPattern = get_emoji_pattern()
tamilPattern = re.compile(r'^[\u0B80-\u0BFF\s\'\"?.{}(),0-9]+$')
ds = load_dataset("uonlp/CulturaX", "ta", split="train", streaming=True)
print("streaming dataset...")
for data in ds:
    text = data["text"]
    text = emojiPattern.sub(r'', text)
    lines = text.split("\n")
    for line in lines:
        if tamilPattern.match(line):
            line = re.sub(r'\.{2,}', '.', line)
            size += len(line.encode("utf8"))
            file.write(line + "\n")
    totSize = (id * FILE_SIZE * GB_TO_BYTES) + size
    if size > FILE_SIZE * GB_TO_BYTES:
        size = 0
        file.close()
        id += 1
        file = open(f"data/{id}.txt", 'w', encoding="utf8")
        print(f"total_size: {totSize} with file_id: {id}")
    if totSize >= DATASET_SIZE * GB_TO_BYTES: break
file.close()