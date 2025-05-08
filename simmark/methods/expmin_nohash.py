import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def get_xis(seed, vocab_size, n):
    rng = np.random.default_rng(seed)
    xis = rng.random((n, vocab_size))
    return xis

def top_p_sampling(probs, xi, top_p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens where cumulative probability exceeds top_p
    cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=False).item()

    # Create new probs and xi
    top_p_sorted_indices = sorted_indices[:cutoff_index+1]
    top_p_probs = probs[top_p_sorted_indices]
    top_p_xi = xi[top_p_sorted_indices.numpy()]

    next_token = torch.argmin(-np.log(top_p_xi) / top_p_probs)

    # Map back to original indices
    return sorted_indices[next_token].item()

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
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            probs = logits[b].softmax(dim=-1)         
            next_token = top_p_sampling(probs, xi, 0.9)
            
            # Modify logits to enforce next token selection
            logits[b, :] = -1e5
            logits[b, next_token] = 1e5

        self.i += 1
        
        return logits  

from scipy.stats import gamma

def expmin_nohash_detect(text, config):
    vocab_size = config['vocab_size']
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()

    n = config['n']
    xis = get_xis(config['seed'], vocab_size, n)
    n_runs = config['n_runs']
    
    test_result = test_statistic(ids, xis, len(ids))
    p_value = 0
    for run in range(n_runs):
        rng = np.random.default_rng()
        xis = rng.random((n, vocab_size))
        null_result = test_statistic(ids, xis, len(ids))
        # assuming lower test values indicate presence of watermark
        p_value += (null_result <= test_result).astype(float) / n_runs

    print(f"p-value: {p_value}")
    return p_value

def test_statistic(ids, xis, k):
    n = xis.shape[0]
    min_cost = float('inf')
    for i in range(len(ids)-k+1):
        min_cost_k = float('inf')

        for j in range(n):
            cost = 0
            for idx in range(k):
                cost += -np.log(xis[(i+j+idx)%n,ids[idx].item()])
            min_cost_k = min(min_cost_k, cost)
        min_cost = min(min_cost_k, min_cost)
    return min_cost
