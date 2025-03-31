import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def get_xis(seed, vocab_size, n):
    rng = np.random.default_rng(seed)
    xis = rng.random((n, vocab_size))
    return xis

class ExpMinNoHashProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.vocab_size = generation_config['vocab_size']
        self.seed = generation_config['seed']
        self.n = generation_config['n']
        self.xis = get_xis(self.seed, self.vocab_size, self.n)
        self.i = 1
        rng = np.random.default_rng()
        self.tau = rng.integers(self.n)

    def forward(self, input_ids, logits):
        xi = self.xis[(self.i+self.tau)%self.n,:]
        probs = logits[0].softmax(dim=-1)         
        next_token = torch.argmin(-np.log(xi) / probs) 
        
        # Modify logits to enforce next token selection
        logits[0, :] = -1e5
        logits[0, next_token] = 1e5

        self.i += 1
        
        return logits  

from scipy.stats import gamma

def expmin_nohash_detect(text, config):
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()

    n = config['n']
    xis = get_xis(config['seed'], config['vocab_size'], n)
    min_cost = float('inf')

    for j in range(n):
        cost = 0
        for i in range(len(ids)):
            cost += -np.log(xis[(j+i)%n,ids[i].item()])
        min_cost = min(min_cost, cost)

    shape = len(ids)
    rate = 1
    p_value = gamma.cdf(min_cost, a=shape, scale=1/rate)

    print(f"Detection cost: {min_cost}, shape: {shape}, rate: {rate}, p-value: {p_value}")
    return p_value

