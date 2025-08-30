import torch
import numpy as np
from scipy.stats import binom

def generate_green(vocab_size, seed, input_ids):
    # print(f"seed: {seed} input_ids: {input_ids.sum().item()}")
    rng = np.random.default_rng(seed*input_ids.sum().item())
    return rng.integers(low=0, high=2, size=vocab_size)

# A function that adjusts the logits so that tokens in the green list have a higher value compared to tokens in the red
# Used within GreenPreferenceProcessor
def adjust_logits(logits, vocab_size, seed, input_ids, bias_factor = 2):
    for i in range(len(input_ids)):
        green_list = generate_green(vocab_size, seed, input_ids[i,:])
        green_tensor = torch.tensor(green_list, dtype=logits.dtype, device=logits.device)
        logits[i] = logits[i] + bias_factor * green_tensor
        probs = logits[i].softmax(dim=-1)
        next_token = top_p_sampling(probs, 0.9)
        logits[i,:] = 1e-5
        logits[i,next_token] = 1e5
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
class SoftRedProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.vocab_size = generation_config['vocab_size']
        self.seed = generation_config['seed']
        self.n_gram_size = generation_config['n_gram']
        # Generate binary green vector
    def forward(self, input_ids, scores):
        # print(f"ids of generation: {input_ids[0]}")

        updated_logits = adjust_logits(scores, self.vocab_size, self.seed, input_ids[:, -(self.n_gram_size - 1):])
  
        return updated_logits

# Detects whether text was likely generated using red/green list technique

def softred_detect(text, detection_config):
    vocab_size = detection_config['vocab_size']
    seed = detection_config['seed']
    n_gram_size = detection_config['n_gram']

    num_green = 0
    ids = detection_config['tokenizer'].encode(text, return_tensors="pt").squeeze().to(detection_config['model'].device)
    # print(F"ids: {ids}")
    for i in range(n_gram_size-1, len(ids)):
        green = generate_green(vocab_size, seed, ids[i-n_gram_size+1:i])
        if green[ids[i]]:
            num_green += 1

    shape = len(ids) - n_gram_size + 1
    # Probability we would expect num_green or more from len(ids) 1/2 trials
    p_val = 1-binom.cdf(num_green, shape, 1/2)

    return p_val