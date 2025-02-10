import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

def simhash_detect(tokenizer, model, vocab_size, text, hash_len=10, k=10, b=32): # MISSING VECTORS AS A PARAM
    embedding_dimension = model.config.hidden_size

    ids = tokenizer.encode(text, return_tensors="pt").squeeze() # ok
    with torch.no_grad(): # ok
        embeddings = model.model.decoder.embed_tokens(ids) # ok

    cost = 0 # ok
    denominator = len(ids) - hash_len  # ok

    for i in range(hash_len, len(ids)): # ok
        input_vector = embeddings[i-hash_len:i].mean(dim=0) # ok
        # input_vector = input_vector.reshape(-1)  # different
        input_vector = input_vector.reshape(embedding_dimension) # added

        min_cost = float('inf') # different

        for ell in range(k): # different 
           
            # NOT REALLY SAMPLING FROM SEED AND ELL ???
            r_vectors = torch.randn((b, embedding_dimension)) # different (takes vectors as input) (is this ok??)
            # r_vectors = torch.randn((b, embedding_dimension)) * (ell + 1) # IS THIS A BETTER OPTION?

            projections = torch.matmul(r_vectors, input_vector) # ok
            simhash_binary = (projections > 0).int() # ok
            simhash_seed = int("".join(map(str, simhash_binary.tolist())), 2) # ok
            
            torch.manual_seed(simhash_seed) # ok
            xi = torch.rand(vocab_size) 

            # current_cost = torch.log(1 - xi[ids[i]]).item() # added
            # current_cost = -torch.log(xi[ids[i]]).item() if xi[ids[i]] < 1 else 0  # different
            current_cost = torch.log(1 - xi[ids[i]]).item() if xi[ids[i]] < 1 else 0 # best ?
            min_cost = min(min_cost, current_cost) # different

        cost += min_cost / denominator # adds the negative but since we do 1-xi i think ok just check they are equal

    return cost # ok
