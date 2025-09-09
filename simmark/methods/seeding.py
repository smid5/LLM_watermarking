import torch
import numpy as np
import hashlib

def to_bytes(x):
    if isinstance(x, int):
        return x.to_bytes((x.bit_length() + 7) // 8 or 1, byteorder='big')
    elif isinstance(x, np.integer):
        return x.tobytes()
    # Torch scalar tensor
    elif isinstance(x, torch.Tensor) and x.dim() == 0:
        return int(x.item()).to_bytes((x.bit_length() + 7) // 8 or 1, byteorder='big')
    # List / NumPy array / Torch tensor (vector)
    if isinstance(x, (list, np.ndarray, torch.Tensor)):
        return b"".join(to_bytes(int(i)) for i in x)
    else:
        print(x)
        print(type(x))
        raise TypeError("Unsupported type for to_bytes")

def simhash_seed(input_ids, prior_tokens, tokenizer, transformer_model, hash_idx, seed, k, b):
    with torch.no_grad():  
        input_text = tokenizer.decode(input_ids[-prior_tokens:], skip_special_tokens=True)
    input_vector = transformer_model.encode(input_text)
    # Use seed and ell to sample b Gaussian vectors r_1, …, r_b in R^d
    rng = np.random.default_rng(hash_idx + k * seed)
    embed_dim = input_vector.shape[0] #384
    random_vectors = rng.standard_normal((b, embed_dim))

    # Apply SimHash to input_vector
    projections = random_vectors @ input_vector
    binary = (projections > 0).astype(int)
    simhash_seed = int(
        hashlib.sha256(to_bytes(hash_idx + k*seed) + to_bytes(binary))
        .hexdigest(),
    16)

    return simhash_seed % 2**32

def normal_seed(input_ids, prior_tokens, hash_idx, seed):
    prior_ids = input_ids[-prior_tokens:].sum().item()
    normal_seed = int(hashlib.sha256(to_bytes(hash_idx) + to_bytes(seed) + to_bytes(prior_ids)).hexdigest(), 16)
    return normal_seed % 2**32