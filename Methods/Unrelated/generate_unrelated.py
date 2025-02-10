import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

# model_name = "facebook/opt-1.3b"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# Generates text using the pretrained model without any watermarking techniques
def generate(tokenizer, model, vocab_size, num_tokens, text):
    input_ids=tokenizer.encode(text, return_tensors="pt")

    outputs = model.generate(
        input_ids,
        max_new_tokens=num_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)