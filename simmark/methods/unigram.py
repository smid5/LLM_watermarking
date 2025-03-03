import torch
import numpy as np
from scipy.stats import binom

def generate_green(vocab_size, seed):
    np.random.seed(seed)
    return np.random.choice([0, 1], vocab_size, p=[0.5, 0.5])

# A function that adjusts the logits so that tokens in the green list have a higher value compared to tokens in the red
# Used within GreenPreferenceProcessor
def adjust_logits(logits, green_list, bias_factor = 6):
    return logits + bias_factor * green_list

# A Logits Processor that adjusts the logits so tokens in the green list are favored
class UnigramProcessor(torch.nn.Module):
    def __init__(self, generation_config):
        super().__init__()
        # Generate binary green vector
        self.green = generate_green(generation_config['vocab_size'], generation_config['seed'])
    def forward(self, input_ids, scores):
        return adjust_logits(scores[0], self.green).unsqueeze(0)

# Detects whether text was likely generated using red/green list technique

def unigram_detect(text, detection_config):
    green = generate_green(detection_config['vocab_size'], detection_config['seed'])

    num_green = 0
    ids = detection_config['tokenizer'].encode(text, return_tensors="pt")[0]
    for token in ids:
        if green[token]:
            num_green += 1

    # Probability we would expect num_green or more from len(ids) 1/2 trials
    p_val = 1-binom.cdf(num_green, len(ids), 1/2)

    return p_val