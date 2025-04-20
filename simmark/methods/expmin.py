import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def get_xi(prior_ids, hash_idx, seed, vocab_size):
    xi_seed = int(hashlib.sha256(
        bytes(str(hash_idx), 'utf-8') + 
        bytes(str(seed), 'utf-8') + 
        bytes(str(prior_ids), 'utf-8')
    ).hexdigest(), 16) % (2**32 - 1)  # Ensure valid seed range

    rng = np.random.default_rng(xi_seed)
    xi = rng.random(vocab_size)
    return xi

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
        self.prior_tokens = generation_config['prior_tokens']
        self.k = generation_config['k']  # Now self.k exists

    def forward(self, input_ids, logits):
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            prior_ids = input_ids[b, -self.prior_tokens:].sum()
            # Sample hash_idx
            rng = np.random.default_rng()
            hash_idx = rng.integers(self.k)
            xi = get_xi(prior_ids, hash_idx, self.seed, self.vocab_size)
            
            probs = logits[b].softmax(dim=-1)         
            next_token = top_p_sampling(probs, xi, 0.9)
            
            # Modify logits to enforce next token selection
            logits[b, :] = -1e5
            logits[b, next_token] = 1e5
        
        return logits  

from scipy.stats import gamma

def expmin_detect(text, config):
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    #prior_tokens = config['prior_tokens']
    prior_tokens = config['prior_tokens']
    with torch.no_grad():
        embeddings = config['model'].model.decoder.embed_tokens(ids)
    
    avg_cost = 0

    if prior_tokens < len(ids):
        for i in range(prior_tokens, len(ids)):
            prior_ids = ids[i-prior_tokens:i].sum() # If i < prior_tokens, this results in an out-of-bounds slice
            min_cost = float('inf')
            
            # Compute the minimum cost for each hash_idx within the window
            for hash_idx in range(config['k']):
                xi = get_xi(prior_ids, hash_idx, config['seed'], config['vocab_size'])
                cost = -np.log(np.clip(xi[ids[i].item()], 1e-10, 1-(1e-10)))  # Get the cost for the actual token id
                min_cost = min(min_cost, cost)

            avg_cost += min_cost / (len(ids) - prior_tokens)

        shape = len(ids) - prior_tokens
        rate = (len(ids) - prior_tokens) * config['k']
        p_value = gamma.cdf(avg_cost, shape, scale=1/rate)
        print(f"Detection cost: {avg_cost}, p-value: {p_value}")
    else:
        p_value = 0.5
        
    return p_value