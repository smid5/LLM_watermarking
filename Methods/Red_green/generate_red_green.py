import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/opt-1.3b"
vocab_size = 50272
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# A function that adjusts the logits so that tokens in the green list have a higher value compared to tokens in the red
# Used within GreenPreferenceProcessor
def adjust_logits(logits, green_list, bias_factor = 6):
    logits += bias_factor * green_list
    return logits

# A Logits Processor that adjusts the logits so tokens in the green list are favored
class GreenPreferenceProcessor(torch.nn.Module):
    def __init__(self, green_list):
        super().__init__()
        self.green_list = green_list
    def forward(self, input_ids, scores):
        return adjust_logits(scores[0], self.green_list).unsqueeze(0)

# Generates text using the red/green list technique
def red_green_generate(vocab_size, num_tokens, text, seed=42):
    torch.manual_seed(seed)
    green_list = torch.randint(2, (vocab_size,), dtype=torch.bool) #chooses 0 or 1
    logits_processor = LogitsProcessorList([GreenPreferenceProcessor(green_list)])
    input_ids=tokenizer.encode(text, return_tensors="pt").to(device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=num_tokens,
        logits_processor=logits_processor, # adjusts the logits here
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

