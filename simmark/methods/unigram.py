import torch
import numpy as np
from scipy.stats import binom

def generate_green(vocab_size, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=2, size=vocab_size)

# A function that adjusts the logits so that tokens in the green list have a higher value compared to tokens in the red
# Used within GreenPreferenceProcessor
def select_next_token(logits, green_list, bias_factor = 2):
    green_tensor = torch.tensor(green_list, dtype=logits.dtype, device=logits.device)
    logits = logits + bias_factor * green_tensor
    probs = logits.softmax(dim=-1)
    next_token = top_p_sampling(probs, 0.9)
    return next_token

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
class UnigramProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        # Generate binary green vector
        self.green = generate_green(generation_config['vocab_size'], generation_config['seed'])
    def forward(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            next_token = select_next_token(scores[b], self.green)
            scores[b,:] = 1e-5
            scores[b,next_token] = 1e5
        return scores

# Detects whether text was likely generated using red/green list technique

def unigram_detect(text, detection_config):
    green = generate_green(detection_config['vocab_size'], detection_config['seed'])

    num_green = 0
    ids = detection_config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    for token in ids:
        if green[token]:
            num_green += 1

    # Probability we would expect num_green or more from len(ids) 1/2 trials
    p_val = 1-binom.cdf(num_green, len(ids), 1/2)

    return p_val