import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash
from detection_simhash import simhash_detect_with_permutation  # Adjusted to the correct import

# Main example: Text generation with watermarking and detection
def example_with_detection():
    # Model and tokenizer setup
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Parameters
    seed = None  # Seed for reproducibility
    vocab_size = tokenizer.vocab_size
    n = 256   # Sampling parameter (unused here)
    m = 50    # Number of tokens to generate

    # Set random seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)  # Set random seed for reproducibility if seed is given
    # torch.manual_seed(seed)

    # Generate initial tokens (starting from an empty prompt)
    prompts = tokenizer("", return_tensors="pt").input_ids  # Empty context to start generation

    # Ranges for k and b
    k_values = [10, 20, 30, 50, 80, 100, 150, 180, 200]  # Number of hash functions
    b_values = [10, 16, 20, 30, 40, 50]      # Bits per hash

    results = []

    for k in k_values:
        for b in b_values:
            print(f"Testing with k={k}, b={b}")

            # Step 1: Generate tokens with SimHash watermarking
            generated_tokens = generate_with_simhash(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                vocab_size=vocab_size,
                n=n,
                m=m,
                # seeds=[seed],
                k=k,
                b=b,
            )
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print("Generated Tokens:", generated_text)

            # Step 2: Detection on the generated sequence
            # Perform detection and compute p-value for the whole sequence
            if seed is not None:
                torch.manual_seed(seed)  # Set random seed for reproducibility if seed is given
            p_value, result, observed_cost = simhash_detect_with_permutation(
                context=generated_text,  # Use the whole generated text as context
                observed_sequence=generated_tokens,  # Use the whole sequence of tokens
                vocab_size=vocab_size,
                k=k,
                b=b,
                # seed=seed,
                model=model,
                tokenizer=tokenizer
            )

            # Store results
            results.append((k, b, p_value))

            # Output results
            print(f"P-value: {p_value:.4f}, Result: {result}, Observed Cost: {observed_cost}")

    # Prepare data for heatmap
    k_indices = {k: i for i, k in enumerate(k_values)}
    b_indices = {b: i for i, b in enumerate(b_values)}
    heatmap = np.full((len(k_values), len(b_values)), np.nan)

    for k, b, p_value in results:
        heatmap[k_indices[k], b_indices[b]] = p_value

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="viridis", aspect="auto")
    plt.colorbar(label="P-value")
    plt.xticks(ticks=np.arange(len(b_values)), labels=b_values)
    plt.yticks(ticks=np.arange(len(k_values)), labels=k_values)
    plt.xlabel("b (Bits per Hash)")
    plt.ylabel("k (Number of Hash Functions)")
    plt.title("Heatmap of P-values for Different k and b Values")
    plt.savefig("Image_outputs/Heatmap of P-values for Different k and b Values.png")
    plt.show()

if __name__ == "__main__":
    example_with_detection()
