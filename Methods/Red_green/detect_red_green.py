import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/opt-1.3b"
vocab_size = 50272
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Detects whether text was likely generated using red/green list technique
from scipy.stats import binom

def red_green_detect(vocab_size, text, seed=42):
    torch.manual_seed(seed)
    green = torch.randint(2, (vocab_size,), dtype=torch.bool)

    num_green = 0
    ids = tokenizer.encode(text, return_tensors="pt")[0]
    for token in ids:
        if green[token]:
            num_green += 1

    # Probability we would expect num_green or more from len(ids) 1/2 trials
    p_val = 1-binom.cdf(num_green, len(ids), 1/2)  # Survival function (1 - CDF)

    return p_val