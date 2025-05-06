import torch
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def simhash(input_vector, hash_idx, vocab_size, seed, k, b):
    # Use seed and ell to sample b Gaussian vectors r_1, …, r_b in R^d
    rng = np.random.default_rng(hash_idx + k * seed)
    embed_dim = input_vector.shape[0] #384
    random_vectors = rng.standard_normal((b, embed_dim))

    # Apply SimHash to input_vector
    projections = random_vectors @ input_vector
    binary = (projections > 0).astype(int)
    simhash_seed = int(
        hashlib.sha256(bytes(hash_idx + k*seed) + bytes(binary))
        .hexdigest(),
    16)

    # Use simhash_seed to sample xi ~ Unif[(0,1)^vocab size]
    rng = np.random.default_rng(simhash_seed % 2**32)
    xi = rng.random(vocab_size)

    return xi

def simmark_detection_cost(text, config):
    ids = config['tokenizer'].encode(text, return_tensors="pt").squeeze()
    transformer_model = SentenceTransformer(config['transformer_model'])

    avg_cost = 0

    for i in range(1, len(ids)):
        input_text = config['tokenizer'].decode(ids[:i])
        input_vector = transformer_model.encode(input_text)
        min_cost = float('inf')
        for hash_idx in range(config['k']):
            xi = simhash(input_vector, hash_idx, config['vocab_size'], config['seed'], config['k'], config['b'])
            cost = -np.log(xi[ids[i]])
            min_cost = min(min_cost, cost)

        avg_cost += min_cost / (len(ids) - 1)

    return avg_cost

def simmark_empirical_detect(text, config, read_file = "data/simmark_emp_null_costs.npy"):
    null_costs = np.load(read_file)

    detection_cost = simmark_detection_cost(text, config)

    empirical_p = np.mean([cost <= detection_cost for cost in null_costs])
    print(f"Empirical p-value: {empirical_p}")
    return empirical_p

def simmark_null_cost(config, output_file = "data/simmark_emp_null_costs.npy"):
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    samples = [sample['highlights'] for sample in dataset if 80 <= len(sample['highlights'].split()) <= 120]
    print(f"samples size = {len(samples)}")
    print(samples[0])
    null_costs = [simmark_detection_cost(sample, config) for sample in samples]

    with open("data/cnntext.txt", "w") as f:
        for sample in samples:
            f.write(sample.strip().replace('\n', ' ') + "\n")
    np.save(output_file, np.array(null_costs))
