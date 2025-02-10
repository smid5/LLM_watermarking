import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash, SimHashWatermark
from detection_simhash import simhash_detect_with_permutation

# Encoder function: Generates embeddings from the input text
# def simple_encoder(text, model, tokenizer):
#     """
#     Encoder function: Converts input text into embeddings using the model's last hidden state.
#     """
#     inputs = tokenizer(text, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#         embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze()  # Mean pooling
#     return embeddings

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

# Function to generate unrelated and unwatermarked text dynamically
def generate_unrelated_text(model, tokenizer, seed):
    torch.manual_seed(seed)
    random_prompt = """In a galaxy far away, a new adventure begins."""  # Example unrelated context
    prompts = tokenizer(random_prompt, return_tensors="pt").input_ids
    outputs = model.generate(prompts, max_length=50, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    base_seed = 42  # Base seed for reproducibility
    vocab_size = tokenizer.vocab_size
    n = 256   # Sampling parameter (unused here)
    m = 50    # Number of tokens to generate
    num_samples = 1  # Number of samples per case

    # Input context
    context = "Once upon a time, in a faraway land, a boy's heart's aching."
    prompts = tokenizer(context, return_tensors="pt").input_ids

    # Collect min_costs for each case
    watermarked_costs = []
    paraphrased_costs = []
    unrelated_costs = []

    # Open a file to write outputs
    with open("text_outputs.txt", "w") as output_file:
        for i in range(num_samples):
            # Generate a unique seed for each sample
            seed = base_seed + i

            # Generate watermarked text
            watermarked_tokens = generate_with_simhash(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                vocab_size=vocab_size,
                n=n,
                m=m,
                seeds=[seed],
                k=k,
                b=b,
            )
            watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
            watermarked_costs.append(calculate_min_cost(watermarked_text, model, tokenizer, simple_encoder, d, k, b, seed))

            # Generate paraphrased text
            paraphrased_text = paraphrase_text(watermarked_text)
            paraphrased_costs.append(calculate_min_cost(paraphrased_text, model, tokenizer, simple_encoder, d, k, b, seed))

            # Generate unrelated and unwatermarked text
            unrelated_text = generate_unrelated_text(model, tokenizer, seed)
            unrelated_costs.append(calculate_min_cost(unrelated_text, model, tokenizer, simple_encoder, d, k, b, seed))

            # Print and write the generated texts
            output = (
                f"ITERATION {i + 1}\n"
                f"Watermarked text: {watermarked_text}\n"
                f"Paraphrased text: {paraphrased_text}\n"
                f"Unrelated text: {unrelated_text}\n\n"
            )
            print(output)
            output_file.write(output)

    # Compute average min_costs
    avg_watermarked = np.mean(watermarked_costs)
    avg_paraphrased = np.mean(paraphrased_costs)
    avg_unrelated = np.mean(unrelated_costs)

    # Print the averages
    print("Average Min Cost Values:")
    print(f"Watermarked Text: {avg_watermarked:.4f}")
    print(f"Paraphrased Text: {avg_paraphrased:.4f}")
    print(f"Unrelated Text: {avg_unrelated:.4f}\n")

    # Write the averages to the file
    with open("text_outputs.txt", "a") as output_file:
        output_file.write("Average Min Cost Values:\n")
        output_file.write(f"Watermarked Text: {avg_watermarked:.4f}\n")
        output_file.write(f"Paraphrased Text: {avg_paraphrased:.4f}\n")
        output_file.write(f"Unrelated Text: {avg_unrelated:.4f}\n\n")

    # Plot bar chart of average min_costs
    plt.figure(figsize=(10, 6))
    categories = ["Watermarked Text", "Paraphrased Text", "Unrelated Text"]
    averages = [avg_watermarked, avg_paraphrased, avg_unrelated]
    
    plt.bar(categories, averages, color=["blue", "green", "red"])
    plt.xlabel("Text Type")
    plt.ylabel("Average Min Cost")
    plt.title("Average Min Cost for Different Text Types")
    plt.grid(axis="y")
    plt.show()

if __name__ == "__main__":
    run_experiment()
