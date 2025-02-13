import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib

def simhash(input_vector, hash_idx, vocab_size, seed, k, b):
    # Use seed and ell to sample b Gaussian vectors r_1, …, r_b in R^d
    np.random.seed(hash_idx + k * seed)
    embed_dim = input_vector.shape[0]
    random_vectors = np.random.randn(b, embed_dim)

    # Apply SimHash to input_vector
    projections = random_vectors @ input_vector.detach().numpy()
    binary = (projections > 0).astype(int)
    simhash_seed = int(
        hashlib.sha256(bytes(seed) + bytes(binary))
        .hexdigest(),
    16)

    # Use simhash_seed to sample xi ~ Unif[(0,1)^vocab size]
    np.random.seed(simhash_seed)
    xi = np.random.rand(vocab_size)

    return xi


class SimMarkProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.model = generation_config['model']
        self.embedding_dimension = self.model.config.hidden_size
        self.vocab_size = generation_config['vocab_size']
        self.k = generation_config['k']
        self.b = generation_config['b']
        self.seed = generation_config['seed']
        self.prior_tokens = generation_config['prior_tokens']

    def forward(self, input_ids, logits):
        # Step 1: Embed context using encoder into vector v in R^d
        with torch.no_grad():  
            embeddings = self.model.model.decoder.embed_tokens(input_ids).squeeze(0)

        input_vector = embeddings[-self.prior_tokens:].mean(dim=0)

        # Sample hash_idx
        hash_idx = np.random.randint(0, self.k)
        # Compute xi using input_vector, hash_idx, and seed
        xi = simhash(input_vector, hash_idx, self.vocab_size, self.seed, self.k, self.b)
 
        probs = logits[0].softmax(dim=-1)         
        next_token = torch.argmin(-np.log(xi) / probs) 
        
        # Modify logits to enforce next token selection
        logits[0, :] = -1e5
        logits[0, next_token] = 1e5
        
        return logits  

from scipy.stats import gamma

def simmark_detect(text, config):
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    prior_tokens = config['prior_tokens']
    with torch.no_grad():
        embeddings = config['model'].model.decoder.embed_tokens(ids)
    
    avg_cost = 0

    for i in range(len(ids)):
        input_vector = embeddings[i-prior_tokens:i].mean(dim=0)
        min_cost = float('inf')
        for hash_idx in range(config['k']):
            xi = simhash(input_vector, hash_idx, config['vocab_size'], config['seed'], config['k'], config['b'])
            cost = -np.log(xi[ids[i]])
            min_cost = min(min_cost, cost)

        avg_cost += min_cost / len(ids)

    shape = len(ids) 
    scale = len(ids) * config['k']
    p_value = gamma.cdf(avg_cost, shape, scale=scale)

    return p_value
