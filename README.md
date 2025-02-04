# tamilGPT

## nanoGPT
Implement Andrej Karptathy's <a href="https://youtu.be/l8pRSuU81PU?si=xJg7CMwFCNscqPL9">nanoGPT</a>

## modded-nanoGPT
<a href="https://github.com/KellerJordan/modded-nanogpt/tree/master">This</a> is a repo trying to train nanoGPT under 3 mins from scratch.

We apply these changes to nanoGPT<br>
* <a href="https://arxiv.org/abs/2104.09864">Rotary embedding</a>
* <a href="https://arxiv.org/abs/2010.04245">Normalize Q,K</a>
* <a href="https://arxiv.org/abs/2109.08668v2">ReLu^2</a>
* Uniform and zero weight initialization
* Skip connections(Encoding/Decoding)
* <a href="https://kellerjordan.github.io/posts/muon/">Muon Optimizer</a>

Now you can train a GPT on a cheap NVIDIA chip under 24 hours.