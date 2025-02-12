import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# ------------------------------
# Things removed/changed from previous code: hash_len removed, introducted r_vectors and ell, adjusted input vectors
# Min_cost now set to -infinity to match pseudocode and min cost done slightly different cause more than 2 vectors now
# ------------------------------

def simhash_detect(tokenizer, model, vocab_size, text, k=10, b=32): # MISSING VECTORS AS A PARAM
    embedding_dimension = model.config.hidden_size 

    # Step 1: Embed context using encoder into vector v in R^d
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()  
    with torch.no_grad(): 
        embeddings = model.model.decoder.embed_tokens(ids) 

    cost = 0 
    # denominator = len(ids) - hash_len  
    denominator = len(ids) # changed to live without hash_len

    for i in range(len(ids)): 
        # Step 1 (continued): Compute mean embedding over the past `hash_len` tokens
        # input_vector = embeddings[i-hash_len:i].mean(dim=0) 

        input_vector = embeddings.mean(dim=0)  # changed to live without hash_len
        input_vector = input_vector.reshape(embedding_dimension)   

        # Step 2: Initialize min_cost = inf
        min_cost = float('inf') # added to match pseudocode

        # Step 3: Iterate over ell in {1,...,k}
        for ell in range(k): # added to match pseudocode
           
            # Step 4: Use (seed, ell) to sample b Gaussian vectors r_1, …, r_b in R^d
            # NOT REALLY SAMPLING FROM SEED AND ELL ???
            torch.manual_seed(0)  
            r_vectors = torch.randn((b, embedding_dimension)) # different (takes vectors as input) (is this ok??)
            # r_vectors = torch.randn((b, embedding_dimension)) * (ell + 1) # IS THIS A BETTER OPTION?

            # Step 5: Instantiate hash function hash_ell with (seed, ell)
            projections = torch.matmul(r_vectors, input_vector) 
            simhash_binary = (projections > 0).int() 

            # Step 6: Compute text_seed = hash_ell(sign(<r_1, v>), …, sign(<r_b, v>))
            simhash_seed = int("".join(map(str, simhash_binary.tolist())), 2) 
            
            # Step 7: Use text_seed to sample xi ~ Unif[(0,1)]^vocab size
            torch.manual_seed(simhash_seed) 
            xi = torch.rand(vocab_size) 

            # Step 8: Compute min_cost = min(min_cost, log(1 - xi_i))
            # current_cost = torch.log(1 - xi[ids[i]]).item() # added
            # current_cost = -torch.log(xi[ids[i]]).item() if xi[ids[i]] < 1 else 0 
            current_cost = -torch.log(xi[ids[i]]).item() # IS THIS THE RIGHT COST?
            min_cost = min(min_cost, current_cost) # adjusted to match pseudocode

        cost += min_cost / denominator 

    avg_cost = cost  # Final avg-cost

    n = len(ids)
    # Compute p-value using Gamma(n, nk)
    shape = n  # Shape parameter
    scale = 1 / (n * k)  # Scale parameter (1 / rate)
    p_value = 1 - gamma.cdf(avg_cost, shape, scale=scale)

    print(p_value)

    return p_value, cost 
