# tamilGPT

## MODEL

### nanoGPT
Implement Andrej Karptathy's <a href="https://youtu.be/l8pRSuU81PU?si=xJg7CMwFCNscqPL9">nanoGPT</a>

### modded-nanoGPT
<a href="https://github.com/KellerJordan/modded-nanogpt/tree/master">This</a> is a repo trying to train nanoGPT under 3 mins from scratch.

We apply these changes to nanoGPT<br>
* <a href="https://arxiv.org/abs/2104.09864">Rotary embedding</a>
* <a href="https://arxiv.org/abs/2010.04245">Normalize Q,K</a>
* <a href="https://arxiv.org/abs/2109.08668v2">ReLu^2</a>
* Uniform weight initialization
* Skip connections(Encoding/Decoding)
* Embeddings to the closest multiple of 128(2^7)

Now you can train a GPT on a cheap NVIDIA chip.

## DATASET
We clean ai4bharat's <a href="https://github.com/AI4Bharat/indicnlp_corpus">dataset</a>

## FAILED EXPERIMENTS
* Zero weight initialization for lm_head and c_proj<br>
    lm_head -> 0 slows the gradient flow to deeper layers. Idk why it has been used in moddedGPT. If you know the answer please reply: https://x.com/_smoke_y/status/1891013258032611364
* <a href="https://kellerjordan.github.io/posts/muon/">Muon Optimizer</a>