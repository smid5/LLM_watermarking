import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def simhash(input_vector, hash_idx, vocab_size, seed, k, b, depth, device):
    g_values = np.empty((depth, vocab_size))
    for i in range(depth):
        # Use seed and ell to sample b Gaussian vectors r_1, …, r_b in R^d
        rng = np.random.default_rng(hash_idx + k * seed)
        embed_dim = input_vector.shape[0] #384
        random_vectors = rng.standard_normal((b, embed_dim))

        # Apply SimHash to input_vector
        projections = random_vectors @ input_vector
        binary = (projections > 0).astype(int)
        simhash_seed = int(
            hashlib.sha256(bytes(hash_idx + k*seed) + bytes(binary) + bytes(i))
            .hexdigest(),
        16) 

        # Use simhash_seed to sample xi ~ Unif[(0,1)^vocab size]
        rng = np.random.default_rng(simhash_seed % 2**32)
        g_values[i,:] = rng.integers(low=0, high=2, size=vocab_size)
    g_tensor = torch.from_numpy(g_values).to(device)

    return g_tensor

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

class SimSynthIDProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.vocab_size = generation_config['vocab_size']
        self.seed = generation_config['seed']
        self.prior_tokens = generation_config['prior_tokens']
        self.depth = generation_config['depth']
        self.model = generation_config['model']
        self.embedding_dimension = self.model.config.hidden_size
        self.k = generation_config['k']
        self.b = generation_config['b']
        self.transformer_model = generation_config['transformer_model']
        self.tokenizer = generation_config['tokenizer']

    def forward(self, input_ids, logits):
        batch_size = input_ids.shape[0]
        g_values = torch.zeros(batch_size, self.depth, self.vocab_size, device=logits.device)
        for batch in range(batch_size):
            # Step 1: Embed context using encoder into vector v in R^d
            with torch.no_grad():  
                input_text = self.tokenizer.decode(input_ids[batch,-8:], skip_special_tokens=True)
            input_vector = self.transformer_model.encode(input_text)

            # Change: Use sentence embedding vector on all prior tokens (not just self.prior_tokens of them)
            # Link in slack for the sentence embedding slibrary

            # Sample hash_idx
            rng = np.random.default_rng()
            hash_idx = rng.integers(self.k)
            # Compute xi using input_vector, hash_idx, and seed
            g_values[batch,:,:] = simhash(input_vector, hash_idx, self.vocab_size, self.seed, self.k, self.b, self.depth, logits.device)
    
        probs = logits.softmax(dim=-1)

        for i in range(self.depth):
            g_values_at_depth = g_values[:,i,:]
            g_mass_at_depth = (g_values_at_depth * probs).sum(dim=-1, keepdims=True)
            probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

        for batch in range(batch_size):
            next_token = top_p_sampling(probs[batch], 0.9)
            logits[batch,:] = 1e-5
            logits[batch,next_token] = 1e5

        return logits

from scipy.stats import rv_discrete, binom
from math import comb

def custom_distribution(n, k, m):
    """
    Custom distribution for the sum of m i.i.d. random variables Y,
    where Y = max(X_1, ..., X_k) and X_i ~ Binomial(n, 0.5)"""
    # CDF of X
    x = np.arange(n+1)
    F = binom.cdf(x, n, 0.5)

    # pmf of Y = max(X_1,...,X_k)
    pY = np.empty(n+1)
    pY[0] = F[0]**k
    pY[1:] = F[1:]**k - F[:-1]**k
    pY = np.clip(pY, 0.0, None)
    pY /= pY.sum()

    # pmf of Z = sum of m iid Y's (convolution)
    pZ = pY.copy()
    for _ in range(m-1):
        pZ = np.convolve(pZ, pY)
        pZ = np.clip(pZ, 0.0, None)
        pZ /= pZ.sum()

    support = np.arange(len(pZ))
    return rv_discrete(name="custom_max_dist", values=(support, pZ))

def simsynthid_detect(text, config):
    device = config['model'].device
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze().to(device)
    transformer_model = config['transformer_model']
    seen_ntuples = set()

    total_max_cost = 0
    num_uniques = 0

    for i in range(1, len(ids)):
        ngram_tokens = tuple(ids[max(0, i-8):i+1].tolist())
        if ngram_tokens in seen_ntuples:
            continue
        seen_ntuples.add(ngram_tokens)
        num_uniques += 1
        input_text = config['tokenizer'].decode(ids[max(0,i-8):i], skip_special_tokens=True)
        input_vector = transformer_model.encode(input_text, device=device)
        max_cost = float('-inf')
        for hash_idx in range(config['k']):
            g_values = simhash(input_vector, hash_idx, config['vocab_size'], config['seed'], config['k'], config['b'], config['depth'], device)
            cost = g_values[:,ids[i]].sum().item()
            max_cost = max(max_cost, cost)

        total_max_cost += max_cost
    
    custom_dist = custom_distribution(config['depth'], config['k'], len(ids)-1)
    p_value = 1 - custom_dist.cdf(total_max_cost)

    print(f"cost: {total_max_cost}, p-value: {p_value}")

    return p_value