import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

# function that randomly replaces "num_modify" tokens in the input text
def modify_text(vocab_size, text, num_modify):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()
    modified_ids = []

    for _ in range(num_modify):
        idx = torch.randint(0, len(ids), (1,))
        while idx[0] in modified_ids:
            idx = torch.randint(0, len(ids), (1,))
        random_token = torch.randint(0, vocab_size, (1,))
        ids[idx] = random_token
        modified_ids.append(idx[0])

    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text