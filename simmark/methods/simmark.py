import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sentence_transformers import SentenceTransformer

def simhash(input_vector, hash_idx, vocab_size, seed, k, b):
    # Use seed and ell to sample b Gaussian vectors r_1, …, r_b in R^d
    rng = np.random.default_rng(hash_idx + k * seed)
    embed_dim = input_vector.shape[0] #384
    random_vectors = rng.standard_normal((b, embed_dim))

    # Apply SimHash to input_vector
    projections = random_vectors @ input_vector
    binary = (projections > 0).astype(int)
    simhash_seed = int(
        hashlib.sha256(bytes(seed) + bytes(binary))
        .hexdigest(),
    16)

    # Use simhash_seed to sample xi ~ Unif[(0,1)^vocab size]
    rng = np.random.default_rng(simhash_seed % 2**32)
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

class SimMarkProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.model = generation_config['model']
        self.embedding_dimension = self.model.config.hidden_size
        self.vocab_size = generation_config['vocab_size']
        self.k = generation_config['k']
        self.b = generation_config['b']
        self.seed = generation_config['seed']
        self.transformer_model = SentenceTransformer(generation_config['transformer_model'])
        self.tokenizer = generation_config['tokenizer']

    def forward(self, input_ids, logits):
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            # Step 1: Embed context using encoder into vector v in R^d
            with torch.no_grad():  
                input_text = self.tokenizer.decode(input_ids[b])
            # if input_text == "</s>Once upon a":
            #     print(f"The first ten indices of logits for prompt \"{input_text}\": {logits[b,:10]}")
            input_vector = self.transformer_model.encode(input_text)

            # Change: Use sentence embedding vector on all prior tokens (not just self.prior_tokens of them)
            # Link in slack for the sentence embedding slibrary

            # Sample hash_idx
            rng = np.random.default_rng()
            hash_idx = rng.integers(self.k)
            # Compute xi using input_vector, hash_idx, and seed
            xi = simhash(input_vector, hash_idx, self.vocab_size, self.seed, self.k, self.b)
    
            probs = logits[b].softmax(dim=-1)         
            next_token = top_p_sampling(probs, xi, 0.9)
            # next_word = self.tokenizer.decode(next_token)
            # print(next_word)
            
            # Modify logits to enforce next token selection
            logits[b, :] = -1e5
            logits[b, next_token] = 1e5
        
        return logits  

from scipy.stats import gamma

def simmark_detect(text, config):
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    transformer_model = SentenceTransformer(config['transformer_model'])

    avg_cost = 0

    for i in range(1, len(ids)):
        input_text = config['tokenizer'].decode(ids[:i])
        input_vector = transformer_model.encode(input_text)
        min_cost = float('inf')
        for hash_idx in range(config['k']):
            xi = simhash(input_vector, hash_idx, config['vocab_size'], config['seed'], config['k'], config['b'])
            cost = -np.log(xi[ids[i]])
            min_cost = min(min_cost, cost)

        avg_cost += min_cost / (len(ids) - 1)

    shape = len(ids) - 1
    rate = (len(ids) - 1) * config['k']
    p_value = gamma.cdf(avg_cost, shape, scale=1/rate)

    print(f"Detection cost: {avg_cost}, p-value: {p_value}")

    return p_value