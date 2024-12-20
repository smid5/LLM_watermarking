import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash, SimHashWatermark
from detection_simhash import simhash_detect_with_permutation

# Encoder function: Generates embeddings from the input text
def simple_encoder(text, model, tokenizer):
    """
    Encoder function: Converts input text into embeddings using the model's last hidden state.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze()  # Mean pooling
    return embeddings

# Function to calculate min_cost for a given text
def calculate_min_cost(text, model, tokenizer, encoder, d, k, b, seed):
    torch.manual_seed(seed)
    embeddings = encoder(text, model, tokenizer)
    # Simulate min_cost calculation (example function, replace with your implementation)
    return embeddings.norm().item()  # Replace with actual min_cost logic

# Function to generate paraphrased text (placeholder)
def paraphrase_text(text):
    # Placeholder for paraphrasing logic
    return text.replace("a", "e")  # Simple example, replace with a real paraphrasing algorithm

# Main experiment function
def run_experiment():
    # Model and tokenizer setup
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Parameters
    d = 2048  # Embedding dimensionality
    k = 100   # Number of hash functions
    b = 30    # Bits per hash
    seed = 42  # Seed for reproducibility
    torch.manual_seed(seed)

    # Input text samples
    watermarked_text = "Once upon a time, in a faraway land, a boy's heart's aching."
    paraphrased_text = paraphrase_text(watermarked_text)
    unrelated_text = "The quick brown fox jumps over the lazy dog."  # Unrelated and unwatermarked

    # Calculate min_cost for each case
    min_costs = {
        "Watermarked Text": calculate_min_cost(watermarked_text, model, tokenizer, simple_encoder, d, k, b, seed),
        "Paraphrased Watermarked Text": calculate_min_cost(paraphrased_text, model, tokenizer, simple_encoder, d, k, b, seed),
        "Unrelated and Unwatermarked Text": calculate_min_cost(unrelated_text, model, tokenizer, simple_encoder, d, k, b, seed),
    }

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.bar(min_costs.keys(), min_costs.values(), color=["blue", "green", "red"])
    plt.xlabel("Text Type")
    plt.ylabel("Average Min Cost")
    plt.title("Histograms of Average Min Cost for Different Text Types")
    plt.grid(axis="y")
    plt.show()

if __name__ == "__main__":
    run_experiment()
