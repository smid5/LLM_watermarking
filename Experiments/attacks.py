import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

# function that randomly replaces "num_modify" tokens in the input text
# def modify_text(tokenizer, vocab_size, text, num_modify):
#     ids = tokenizer.encode(text, return_tensors="pt").squeeze()
#     # print(len(ids))
#     modified_ids = []

#     for _ in range(num_modify):
#         idx = torch.randint(0, len(ids), (1,))
#         while idx[0] in modified_ids:
#             # print("modified ids:" + str(modified_ids))
#             idx = torch.randint(0, len(ids), (1,))
#             modify_indices = torch.tensor(np.random.choice(len(ids), num_modify, replace=False))

#             # print("idx: " + str(idx[0]))
#         random_token = torch.randint(0, vocab_size, (1,))
#         ids[idx] = random_token
#         modified_ids.append(idx[0])

#     text = tokenizer.decode(ids, skip_special_tokens=True)
    
#     return text
    
def modify_text(tokenizer, vocab_size, text, num_modify):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()
    modified_ids = []

    num_modify = min(num_modify, len(ids))  

    for _ in range(num_modify):
        idx = torch.randint(0, len(ids), (1,))
        while idx[0] in modified_ids:
            idx = torch.randint(0, len(ids), (1,))  # Keep searching for a new index

        random_token = torch.randint(0, vocab_size, (1,))
        ids[idx] = random_token
        modified_ids.append(idx[0])

    text = tokenizer.decode(ids, skip_special_tokens=True)
    
    return text
