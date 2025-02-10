import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/opt-1.3b"
vocab_size = 50272
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Detects whether text was likely generated using exponential minimum sampling technique
def exp_detect(vocab_size, text, hash_len=10):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()
    cost = 0
    denominator = len(ids) - hash_len

    for i in range(hash_len, len(ids)):
        seed1 = torch.prod(ids[i-hash_len:i])
        seed2 = seed1//3
        torch.manual_seed(seed1)
        x1 = torch.rand(vocab_size) # Unif([0,1]^vocab_size)
        torch.manual_seed(seed2)
        x2 = torch.rand(vocab_size) # Unif([0,1]^vocab_size)
        if x1[ids[i]] < x2[ids[i]]: # choose the x_vector that has smaller cost
            cost += - torch.log(x2[ids[i]]).item() / denominator
        else:
            cost += - torch.log(x1[ids[i]]).item() / denominator

    return cost