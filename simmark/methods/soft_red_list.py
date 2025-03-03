import torch
import numpy as np
from scipy.stats import binom

def generate_green(vocab_size, seed, input_ids):
    np.random.seed(seed*input_ids.sum().item())
    return np.random.choice([0, 1], vocab_size, p=[0.5, 0.5])

# A function that adjusts the logits so that tokens in the green list have a higher value compared to tokens in the red
# Used within GreenPreferenceProcessor
def adjust_logits(logits, green_list, bias_factor = 6):
    return logits + bias_factor * green_list

# A Logits Processor that adjusts the logits so tokens in the green list are favored
class SoftRedProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        self.vocab_size = generation_config['vocab_size']
        self.seed = generation_config['seed']
        self.n_gram_size = generation_config['n_gram']
        # Generate binary green vector
    def forward(self, input_ids, scores):

        self.green = generate_green(self.vocab_size, self.seed, input_ids[-(self.n_gram_size - 1):])
  
        return adjust_logits(scores, self.green)

# Detects whether text was likely generated using red/green list technique

def softred_detect(text, detection_config):
    vocab_size = detection_config['vocab_size']
    seed = detection_config['seed']
    n_gram_size = detection_config['n_gram']

    num_green = 0
    ids = detection_config['tokenizer'].encode(text, return_tensors="pt")[0]
    for i in range(n_gram_size, len(ids)-1):
        green = generate_green(vocab_size, seed, ids[i-n_gram_size+2:i])
        if green[ids[i+1]]:
            num_green += 1

    # Probability we would expect num_green or more from len(ids) 1/2 trials
    p_val = 1-binom.cdf(num_green, len(ids)-1, 1/2)

    return p_val