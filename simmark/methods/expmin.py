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

class ExpMinProcessor(torch.nn.Module):
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

from scipy.stats import gamma, rv_continuous

# Custom distribution
class MinGammaDist(rv_continuous):
    def __init__(self, alpha, scale, n):
        super().__init__()
        self.alpha = alpha
        self.n = n
        self.scale = scale

    def _pdf(self, x):
        f_x = gamma.pdf(x, a=self.alpha, scale=self.scale)
        F_x = gamma.cdf(x, a=self.alpha, scale=self.scale)
        return self.n * (1 - F_x)**(self.n - 1) * f_x

    def _cdf(self, x):
        F_x = gamma.cdf(x, a=self.alpha, scale=self.scale)
        return 1 - (1 - F_x)**self.n


# def expmin_detect(text, config):
#     avg_cost = 0
#     n = config['n']
#     vocab_size = config['vocab_size']
#     xis = get_xis(config['seed'], vocab_size, n)
#     ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()

#     min_cost = float('inf')
#     for offset in range(n):
#         cost = 0
#         for i in range(len(ids)):
#             cost += -np.log(xis[(i+offset)%n,ids[i].item()])
#         min_cost = min(min_cost, cost)

#     shape = len(ids)
#     rate = 1
#     num_rvs = n
#     min_gamma = MinGammaDist(alpha=shape, scale=1/rate, n=num_rvs)
#     p_value = min_gamma.cdf(min_cost)
#     print(f"Detection cost: {min_cost}, p-value: {p_value}")
        
#     return p_value

def expmin_detect(text, config):
    n = config['n']
    vocab_size = config['vocab_size']
    xis = get_xis(config['seed'], vocab_size, n)
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    k = config['k']

    min_cost = float('inf')
    for i in range(len(ids)-k+1):
        min_cost_i = float('inf')
        for offset in range(n):
            cost = 0
            for j in range(k):
                cost += -np.log(xis[(i+offset+j)%n,ids[i+j].item()])
            min_cost_i = min(min_cost_i, cost)
        min_cost = min(min_cost, min_cost_i)

    shape = k
    rate = 1
    num_rvs = n * (len(ids)-k+1)
    min_gamma = MinGammaDist(alpha=shape, scale=1/rate, n=num_rvs)
    p_value = min_gamma.cdf(min_cost)
    print(f"Detection cost: {min_cost}, p-value: {p_value}")
        
    return p_value