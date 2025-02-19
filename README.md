```
  __                    .__ .__     ________ __________ ___________ 
_/  |_ _____     _____  |__||  |   /  _____/ \______   \\__    ___/ 
\   __\\__  \   /     \ |  ||  |  /   \  ___  |     ___/  |    |    
 |  |   / __ \_|  Y Y  \|  ||  |__\    \_\  \ |    |      |    |    
 |__|  (____  /|__|_|  /|__||____/ \______  / |____|      |____|    
            \/       \/                   \/                        
```

Training a GPT in 4 hours on tamil tokens.

## MODEL

### nanoGPT
Implement Andrej Karptathy's <a href="https://github.com/karpathy/build-nanogpt">nanoGPT</a>

### modded-nanoGPT
<a href="https://github.com/KellerJordan/modded-nanogpt/tree/master">This</a> is a repo trying to train nanoGPT under 3 mins from scratch. Using this repo as a reference, we apply these changes to nanoGPT<br>
* <a href="https://arxiv.org/abs/2104.09864">Rotary embedding</a>
* <a href="https://arxiv.org/abs/2010.04245">Normalize Q,K</a>
* <a href="https://arxiv.org/abs/2109.08668v2">ReLu^2</a>
* Uniform weight initialization
* Skip connections(Encoding/Decoding)
* Embeddings to the closest multiple of 128(2^7)

Now you can train a GPT on a cheap NVIDIA chip.

## GETTING STARTED
Download ai4bharat's <a href="https://github.com/AI4Bharat/indicnlp_corpus">dataset</a>(ta.txt) and place it under ```data/```. Run ```src/clean.py``` and finally ```src/train.py```. Modify batch size based on your VRAM(```src/model.py```).

You can find the weights <a href="https://huggingface.co/smoke-y/tamilGPT">here</a>.

## FAILED EXPERIMENT
* Zero weight initialization for lm_head and c_proj<br>
    lm_head -> 0 slows the gradient flow to deeper layers. Idk why it has been used in moddedGPT. If you know the answer: https://x.com/_smoke_y/status/1891013258032611364