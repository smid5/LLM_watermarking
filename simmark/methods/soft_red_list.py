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
    green_list = np.zeros((len(input_ids), vocab_size))
    for i in range(len(input_ids)):
        green_list[i,:] = generate_green(vocab_size, seed, input_ids[i,:])
    return logits + bias_factor * torch.tensor(green_list)

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
    ids = detection_config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    # print(F"ids: {ids}")
    for i in range(n_gram_size-1, len(ids)):
        green = generate_green(vocab_size, seed, ids[i-n_gram_size+1:i])
        if green[ids[i]]:
            num_green += 1

    shape = len(ids) - n_gram_size + 1
    # Probability we would expect num_green or more from len(ids) 1/2 trials
    p_val = 1-binom.cdf(num_green, shape, 1/2)

    return p_val