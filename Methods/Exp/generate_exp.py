import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/opt-1.3b"
vocab_size = 50272
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# A Logits Processor that adjusts the logits so that the value at i_star has very high value, while other values are close to 0
class ExponentialSamplingProcessor(torch.nn.Module):
    def __init__(self, vocab_size, hash_len=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.hash_len = hash_len
        self.past_input_ids = None

    def forward(self, input_ids, logits):
        self.past_input_ids = input_ids

        # Generate the hash seed from the last 'hash_len' tokens
        if self.past_input_ids.size(1) >= self.hash_len:
            seed1 = torch.prod(self.past_input_ids.squeeze()[-self.hash_len:])
        else:
            seed1 = torch.prod(self.past_input_ids.squeeze())
        seed2 = seed1//3 # Introduces randomness by using 2 x_vectors
        dist = logits[0].softmax(dim=-1)

        torch.manual_seed(seed1)
        x1 = torch.rand(vocab_size)  # Generate randomness in the range [0, 1]^vocab_size
        next_token1 = torch.argmin(-torch.log(dist) / x1)

        torch.manual_seed(seed2)
        x2 = torch.rand(vocab_size)  # Generate randomness in the range [0, 1]^vocab_size
        next_token2 = torch.argmin(-torch.log(dist) / x2)

        torch.manual_seed(0)
        next_token = next_token1 if torch.rand(1) < 0.5 else next_token2 # chooses either token with equal chance
        logits[0, :] = -100000  # close to 0 after applying softmax
        logits[0, next_token] = 100000

        return logits

# Generates text using exponential minimum sampling
def exp_generate(vocab_size, num_tokens, text, hash_len=10):

    logits_processor = LogitsProcessorList([ExponentialSamplingProcessor(vocab_size, hash_len)])
    input_ids=tokenizer.encode(text, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=num_tokens,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)