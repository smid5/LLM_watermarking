import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42) # ok 

class SimHashProcessor(torch.nn.Module):
    def __init__(self, tokenizer, vocab_size, model, k=10, b=16, hash_len=10):
        super().__init__()
        self.vocab_size = vocab_size # ok
        self.model = model # ok
        self.k = k # ok
        self.b = b # ok
        self.hash_len = hash_len # added
        self.embedding_dimension = model.config.hidden_size

    def forward(self, input_ids, logits):
        # Step 1: Embed context using encoder into vector v in R^d
        with torch.no_grad(): # added
            embeddings = self.model.model.decoder.embed_tokens(input_ids) # added

        embeddings = embeddings.squeeze(0) # added
        # input_vector = embeddings[-self.hash_len:].mean(dim=0) # old
        input_vector = embeddings.mean(dim=0)  # changed to match pseudocode

        # Step 2: Sample ell uniformly from {1, ..., k}
        ell = torch.randint(0, self.k, (1,)).item() # ok
        
        # Step 3: Use (seed, ell) to sample b Gaussian vectors r_1, …, r_b in R^d
        r_vectors = torch.randn((self.b, self.embedding_dimension))
        
        # Step 4: Instantiate hash function hash_ell with (seed, ell)
        projections = torch.matmul(r_vectors, input_vector) # ok
        simhash_binary = (projections > 0).int() # ok
        simhash_seed = int("".join(map(str, simhash_binary.tolist())), 2) # ok
        
        # Step 5: Compute text_seed = hash_ell(sign(<r_1, v>), …, sign(<r_b, v>))
        torch.manual_seed(simhash_seed) # ok

        # Step 6: Use text_seed to sample xi ~ Unif[(0,1)]^vocab size
        x = torch.rand(self.vocab_size)  
        
        # Step 7: Evaluate probability distribution p = LLM(context)
        dist = logits[0].softmax(dim=-1) # ok

        # Step 8: Apply exponential minimum sampling to get i* = max_j log(xi_j) / p_j
        next_token = torch.argmin(-torch.log(dist) / x) # ok
        
        # Reset random seed so that other random processes aren't affected
        torch.manual_seed(0) 

        # MISSING A NEXT TOKEN SELECTION THING HERE (ok i think)
        # Modify logits to enforce next token selection
        logits[0, :] = -100000 # added 
        logits[0, next_token] = 100000 # ok
        
        return logits # ok

def simhash_generate(tokenizer, model, vocab_size, num_tokens, text, k=1, b=32, hash_len=10): 
    input_ids = tokenizer.encode(text, return_tensors="pt") # pl
    
    simhash_processor = SimHashProcessor(tokenizer, vocab_size, model, k=k, b=b, hash_len=hash_len) 
    
    logits_processor_list = LogitsProcessorList([simhash_processor]) 

    outputs = model.generate( 
        input_ids,
        max_new_tokens=num_tokens,
        logits_processor=logits_processor_list,
        pad_token_id=tokenizer.eos_token_id
    ) # ok
    return tokenizer.decode(outputs[0], skip_special_tokens=True) # ok
