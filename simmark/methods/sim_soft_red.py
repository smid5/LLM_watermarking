import torch
import numpy as np
from scipy.stats import binom
import hashlib

def simhash_generate_green(input_vector, vocab_size, seed, b):
    # Use seed and ell to sample b Gaussian vectors r_1, …, r_b in R^d
    rng = np.random.default_rng(seed)
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
    return rng.integers(low=0, high=2, size=vocab_size)

def generate_green(vocab_size, seed, input_ids):
    # print(f"seed: {seed} input_ids: {input_ids.sum().item()}")
    rng = np.random.default_rng(seed*input_ids.sum().item())
    return rng.integers(low=0, high=2, size=vocab_size)

# A function that adjusts the logits so that tokens in the green list have a higher value compared to tokens in the red
# Used within GreenPreferenceProcessor
def adjust_logits(logits, vocab_size, seed, input_vector, b, bias_factor = 2):
    green_list = simhash_generate_green(input_vector, vocab_size, seed, b)
    green_tensor = torch.tensor(green_list, dtype=logits.dtype, device=logits.device)
    logits = logits + bias_factor * green_tensor
    probs = logits.softmax(dim=-1)
    next_token = top_p_sampling(probs, 0.9)
    logits[:] = 1e-5
    logits[next_token] = 1e5
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

# A Logits Processor that adjusts the logits so tokens in the green list are favored
class SimSoftRedProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.n_gram_size = generation_config['n_gram']
        self.model = generation_config['model']
        self.embedding_dimension = self.model.config.hidden_size
        self.vocab_size = generation_config['vocab_size']
        self.b = generation_config['b']
        self.seed = generation_config['seed']
        self.transformer_model = generation_config['transformer_model']
        self.tokenizer = generation_config['tokenizer']
        # Generate binary green vector
    def forward(self, input_ids, scores):
        # print(f"ids of generation: {input_ids[0]}")
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            with torch.no_grad():  
                input_text = self.tokenizer.decode(input_ids[b,-(self.n_gram_size - 1):], skip_special_tokens=True)
            input_vector = self.transformer_model.encode(input_text)
            updated_logits = adjust_logits(
                scores[b], self.vocab_size, self.seed, input_vector, self.b
            )
            scores[b] = updated_logits

        return scores

# Detects whether text was likely generated using red/green list technique

def simsoftred_detect(text, detection_config):
    vocab_size = detection_config['vocab_size']
    seed = detection_config['seed']
    n_gram_size = detection_config['n_gram']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = detection_config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    transformer_model = detection_config['transformer_model']
    seen_ntuples = set()

    num_green = 0
    # print(F"ids: {ids}")
    for i in range(n_gram_size-1, len(ids)):
        ngram_tokens = tuple(ids[max(0, i-n_gram_size+1):i+1].tolist())
        if ngram_tokens in seen_ntuples:
            continue
        seen_ntuples.add(ngram_tokens)
        input_text = detection_config['tokenizer'].decode(ids[max(0,i-n_gram_size+1):i], skip_special_tokens=True)
        input_vector = transformer_model.encode(input_text, device=device)
        green = simhash_generate_green(input_vector, vocab_size, seed, detection_config['b'])
        if green[ids[i]]:
            num_green += 1

    shape = len(ids) - n_gram_size + 1
    # Probability we would expect num_green or more from len(ids) 1/2 trials
    p_val = 1-binom.cdf(num_green, shape, 1/2)

    return p_val