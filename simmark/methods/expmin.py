import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib

# def get_xi(prior_ids, hash_idx, seed, vocab_size):
#     xi_seed = int(hashlib.sha256(hash_idx + bytes(seed) + bytes(prior_ids)).hexdigest(), 16)
#     np.random.seed(xi_seed)
#     xi = np.random.rand(vocab_size)
#     return xi

def get_xi(prior_ids, hash_idx, seed, vocab_size):
    xi_seed = int(hashlib.sha256(
        bytes(str(hash_idx), 'utf-8') + 
        bytes(str(seed), 'utf-8') + 
        bytes(str(prior_ids), 'utf-8')
    ).hexdigest(), 16) % (2**32 - 1)  # Ensure valid seed range

    rng = np.random.default_rng(xi_seed)
    xi = rng.random(vocab_size)
    return xi


class ExpMinProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.vocab_size = generation_config['vocab_size']
        self.seed = generation_config['seed']
        self.prior_tokens = generation_config['prior_tokens']
        self.k = generation_config['k']  # Now self.k exists

    def forward(self, input_ids, logits):
        prior_ids = input_ids[0, -self.prior_tokens:].sum()
        # Sample hash_idx
        rng = np.random.default_rng()
        hash_idx = rng.integers(self.k)
        # hash_idx = int(hashlib.sha256(bytes(str(prior_ids), 'utf-8')).hexdigest(), 16) % self.k
        xi = get_xi(prior_ids, hash_idx, self.seed, self.vocab_size)
         
        probs = logits[0].softmax(dim=-1)         
        next_token = torch.argmin(-np.log(xi) / probs) 
        
        # Modify logits to enforce next token selection
        logits[0, :] = -1e5
        logits[0, next_token] = 1e5
        
        return logits  

# from scipy.stats import gamma

# def expmin_detect(text, config):
#     ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()
#     prior_tokens = config['prior_tokens']
#     with torch.no_grad():
#         embeddings = config['model'].model.decoder.embed_tokens(ids)
    
#     avg_cost = 0

#     for i in range(len(ids)):
#         prior_ids = ids[i-prior_tokens:i]
#         min_cost = float('inf')
#         for hash_idx in range(config['k']):            
#             xi = get_xi(prior_ids, hash_idx, config['seed'], config['vocab_size'])
#             cost = -np.log(xi[ids[i]])
#             min_cost = min(min_cost, cost)

#         avg_cost += min_cost / len(ids)

#     shape = len(ids) 
#     scale = len(ids) * config['k']
#     p_value = gamma.cdf(avg_cost, shape, scale=scale)

#     return p_value

from scipy.stats import gamma

def expmin_detect(text, config):
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    #prior_tokens = config['prior_tokens']
    prior_tokens = config['prior_tokens']
    with torch.no_grad():
        embeddings = config['model'].model.decoder.embed_tokens(ids)
    
    avg_cost = 0

    for i in range(prior_tokens, len(ids)):
        prior_ids = ids[i-prior_tokens:i].sum() # If i < prior_tokens, this results in an out-of-bounds slice
        min_cost = float('inf')
        
        # Compute the minimum cost for each hash_idx within the window
        for hash_idx in range(config['k']):
            xi = get_xi(prior_ids, hash_idx, config['seed'], config['vocab_size'])
            cost = -np.log(np.clip(xi[ids[i].item()], 1e-10, 1))  # Get the cost for the actual token id
            min_cost = min(min_cost, cost)

        avg_cost += min_cost / (len(ids) - prior_tokens)

    shape = len(ids) - prior_tokens
    rate = (len(ids) - prior_tokens) * config['k']
    p_value = gamma.cdf(avg_cost, shape, scale=1/rate)

    print(f"Detection cost: {avg_cost}, p-value: {p_value}")
    return p_value

