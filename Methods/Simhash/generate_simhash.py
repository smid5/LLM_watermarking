import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "facebook/opt-1.3b"
vocab_size = 50272
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# GENERALIZATION 

import torch # ok

torch.manual_seed(42) # ok 

embedding_dimension = model.config.hidden_size # ok

class SimHashProcessor(torch.nn.Module):
    def __init__(self, vocab_size, model, k=10, b=16, hash_len=10): # added hash_len
        super().__init__()
        self.vocab_size = vocab_size # ok
        self.model = model # ok
        self.k = k # ok
        self.b = b # ok
        self.hash_len = hash_len # added
        self.past_input_ids = None # added

        self.device = next(model.parameters()).device  # different

    def forward(self, input_ids, logits):
        # MISSING PAST_INPUT_IDS
        self.past_input_ids = input_ids # added

        # with torch.no_grad(): # ok
        #     embeddings = self.model.model.decoder.embed_tokens(input_ids)  # different
        # input_vector = embeddings.mean(1).squeeze(0) # different
        with torch.no_grad(): # added
            embeddings = self.model.model.decoder.embed_tokens(input_ids) # added

        embeddings = embeddings.squeeze(0) # added
        input_vector = embeddings[-self.hash_len:].mean(dim=0) # added

        ell = torch.randint(0, self.k, (1,)).item() # ok
        
        r_vectors = torch.randn((self.b, embedding_dimension)).to(device)
        
        projections = torch.matmul(r_vectors, input_vector) # ok
        simhash_binary = (projections > 0).int() # ok
        simhash_seed = int("".join(map(str, simhash_binary.tolist())), 2) # ok
        
        torch.manual_seed(simhash_seed) # ok
        x = torch.rand(self.vocab_size, device=self.device)  
        
        # dist = logits.softmax(dim=-1) # different and done earlier in other code
        dist = logits[0].softmax(dim=-1) # added

        next_token = torch.argmin(-torch.log(dist) / x) # ok
        
        torch.manual_seed(0) # added
        # logits.fill_(-100000) # different 
        # MISSING A NEXT TOKEN SELECTION THING HERE (ok i think)
        logits[0, :] = -100000 # added 
        logits[0, next_token] = 100000 # ok
        
        return logits # ok

def simhash_generate(vocab_size, num_tokens, text, k=1, b=32, hash_len=10): # missing hash_len here as param (added hash_len)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device) # pl
    
    simhash_processor = SimHashProcessor(vocab_size, model, k=k, b=b, hash_len=hash_len) # ok but just put it in step below for consistency (added hash_len)
    
    logits_processor_list = LogitsProcessorList([simhash_processor]) # ok (just rename to match)

    outputs = model.generate( # ok
        input_ids,
        max_new_tokens=num_tokens,
        logits_processor=logits_processor_list,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True) # ok
