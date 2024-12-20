import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash, SimHashWatermark
from detection_simhash import simhash_detect_with_permutation

# Main example: Text generation with watermarking and detection
def example_with_detection():
    # Model and tokenizer setup
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Parameters
    d = 2048  # Embedding dimensionality
    seed = 42  # Seed for reproducibility
    vocab_size = tokenizer.vocab_size
    n = 256   # Sampling parameter (unused here)
    m = 50    # Number of tokens to generate

    # Set random seeds for reproducibility
    torch.manual_seed(seed)

    # Input context
    context = "Once upon a time, in a faraway land, a boy's heart's aching."
    prompts = tokenizer(context, return_tensors="pt").input_ids

    # Ranges for k and b
    k_values = [50, 100, 150]  # Adjust as needed
    b_values = [20, 30, 40]    # Adjust as needed

    # k_values = [100]
    # b_values = [30]

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
                seeds=[seed],
                k=k,
                b=b,
            )
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print("Generated Tokens:", generated_text)

            # Step 2: Detection on the last generated token
            observed_token = generated_tokens[-1]  # Last generated token

            # Perform detection and compute p-value
            torch.manual_seed(seed)  # Ensure reproducibility
            p_value, result, _ = simhash_detect_with_permutation(
                context=context,
                observed_token=observed_token,
                d=d,
                k=k,
                b=b,
                seed=seed,
                model=model,
                tokenizer=tokenizer
            )

            # Store results
            results.append((k, b, p_value))

            # Output results
            print(f"P-value: {p_value:.4f}")
            print(result)

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
    plt.show()

if __name__ == "__main__":
    example_with_detection()




