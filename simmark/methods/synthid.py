import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib


def get_gvalues(prior_ids, hash_idx, seed, vocab_size, depth):
    g_values = np.empty((depth, vocab_size))
    for i in range(depth):
        g_seed = int(hashlib.sha256(
            bytes(str(hash_idx), 'utf-8') + 
            bytes(str(seed), 'utf-8') + 
            bytes(str(prior_ids), 'utf-8') +
            bytes(str(i), 'utf-8')
        ).hexdigest(), 16) % (2**32 - 1)  # Ensure valid seed range

        np.random.seed(g_seed)
        g_values[i,:] = np.random.rand(vocab_size)
        g_tensor = torch.from_numpy(g_values) 
    return g_tensor

def update_scores(logits, g_values, selfdepth):
    depth, _ = g_values.shape

    assert depth == selfdepth
    probs = logits.softmax(dim=-1)

    for i in range(depth):
        g_values_at_depth = g_values[i,:]
        g_mass_at_depth = (g_values_at_depth * probs).sum(axis=0, keepdims=True)
        probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

    return probs



class SynthIDProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.vocab_size = generation_config['vocab_size']
        self.seed = generation_config['seed']
        self.hash_len = generation_config['hash_len']
        self.k = generation_config['k']  
        self.depth = generation_config['depth']

    def forward(self, input_ids, logits):
        prior_ids = input_ids[0, -self.hash_len:].sum()
        # Sample hash_idx
        hash_idx = np.random.randint(0, self.k)
        g_values = get_gvalues(prior_ids, hash_idx, self.seed, self.vocab_size, self.depth)
         
        probs = update_scores(logits[0], g_values, self.depth)
        
        return probs 

from scipy.stats import binom

def synthid_detect(text, config):
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    
    avg_cost = 0

    # Assuming 'prior_tokens' is defined in 'config' similar to SimMark
    hash_len = config['hash_len']
    for i in range(hash_len, len(ids)):
        # Extract the embeddings for the prior tokens window
        prior_ids = ids[i-hash_len:i].sum() 
        min_cost = float('inf')
        
        # Compute the minimum cost for each hash_idx within the window
        for hash_idx in range(config['k']):
            g_values = get_gvalues(prior_ids, hash_idx, config['seed'], config['vocab_size'], config['depth'])
            cost = g_values[:,ids[i]].sum()  # Get the cost for the actual token id
            min_cost = min(min_cost, cost)

        # Accumulate the average cost normalized by the total number of ids considered
        avg_cost += min_cost / (len(ids) - hash_len)

    # Gamma distribution parameters
    shape = len(ids) - hash_len

    # Calculate the p-value using the gamma cumulative distribution function
    p_value = binom.cdf(avg_cost, shape, 0.5)

    print(f"Detection cost: {avg_cost}, p-value: {p_value}")
    return p_value