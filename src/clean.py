import re
from tqdm import tqdm

data = open("data/ta.txt", "r", encoding="utf8")
cleanedFile = open("data/ta_clean.txt", "w+", encoding="utf8")

englishPattern = re.compile(r'[A-Za-z]')
separatorPattern = re.compile(r'[\u2028\u2029]')
spacePattern = re.compile(r'\s+')
dotPattern = re.compile(r'\.{2,}')

for i,line in enumerate(tqdm(data)):
    if bool(englishPattern.search(line)): continue
    line = separatorPattern.sub(" ", line)
    line = spacePattern.sub(" ", line)
    line = dotPattern.sub(".", line).strip()
    if len(line.split(" ")) < 4: continue
    cleanedFile.write(line + "\n")

data.close()
cleanedFile.close()