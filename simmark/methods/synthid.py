import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib


def get_gvalues(prior_ids, seed, vocab_size, depth):
    g_values = np.empty((depth, vocab_size))
    for i in range(depth):
        g_seed = int(hashlib.sha256(
            bytes(str(seed), 'utf-8') + 
            bytes(str(prior_ids), 'utf-8') +
            bytes(str(i), 'utf-8')
        ).hexdigest(), 16) % (2**32 - 1)  # Ensure valid seed range

        rng = np.random.default_rng(g_seed)
        g_values[i,:] = rng.integers(low=0, high=2, size=vocab_size)
    g_tensor = torch.from_numpy(g_values) 
    return g_tensor

def update_scores(logits, prior_ids, seed, vocab_size, depth):
    batch_size = prior_ids.shape[0]
    g_values = torch.zeros(batch_size, depth, vocab_size)
    for b in range(batch_size):
        g_values[b,:,:] = get_gvalues(prior_ids[b], seed, vocab_size, depth)

    probs = logits.softmax(dim=-1)

    for i in range(depth):
        g_values_at_depth = g_values[:,i,:]
        g_mass_at_depth = (g_values_at_depth * probs).sum(dim=-1, keepdims=True)
        probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

    for b in range(batch_size):
        next_token = top_p_sampling(probs[b], 0.9)
        logits[b,:] = 1e-5
        logits[b,next_token] = 1e5

    return logits

def top_p_sampling(probs, top_p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens where cumulative probability exceeds top_p
    cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=False).item()

    # Set the logits of the tokens beyond top_p to zero
    sorted_probs[cutoff_index+1:] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    sorted_probs = torch.where(
        torch.isfinite(sorted_probs), sorted_probs, torch.tensor(0.0)
    )
    # Sample from the filtered distribution
    next_token = torch.multinomial(sorted_probs, 1)

    # Map back to original indices
    return sorted_indices[next_token].item()

class SynthIDProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.vocab_size = generation_config['vocab_size']
        self.seed = generation_config['seed']
        self.prior_tokens = generation_config['prior_tokens']
        # self.k = generation_config['k']  
        self.depth = generation_config['depth']

    def forward(self, input_ids, logits):
        prior_ids = input_ids[:, -self.prior_tokens:].sum(dim=-1)
        # Sample hash_idx
        # hash_idx = np.random.randint(0, self.k)
         
        scores = update_scores(logits, prior_ids, self.seed, self.vocab_size, self.depth)
        
        return scores

from scipy.stats import binom

def synthid_detect(text, config):
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()

    # Assuming 'prior_tokens' is defined in 'config' similar to SimMark
    prior_tokens = config['prior_tokens']
    # max_cost = float('-inf')

    # Compute the minimum cost for each hash_idx within the window
    # for hash_idx in range(config['k']):
    cost = 0
    for i in range(prior_tokens, len(ids)):
        # Extract the embeddings for the prior tokens window
        prior_ids = ids[i-prior_tokens:i].sum() 
        
        g_values = get_gvalues(prior_ids, config['seed'], config['vocab_size'], config['depth'])
        cost += g_values[:,ids[i]].sum().item()  # Get the cost for the actual token id
    # max_cost = max(max_cost, cost)

    # distribution parameters
    shape = (len(ids) - prior_tokens) * len(g_values)

    score = cost / shape

    # Probability we would expect max_cost or more from shape 1/2 trials
    p_value = 1 - binom.cdf(cost, shape, 0.5)

    print(f"Detection cost: {score}, p-value: {p_value}")
    return p_value