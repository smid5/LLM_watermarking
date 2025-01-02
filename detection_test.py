import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_test import SimHashWatermark

def simple_encoder(text, model, tokenizer):
    """
    Encoder function: Converts input text into embeddings using the model's last hidden state.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze()  # Mean pooling
    return embeddings

# def simhash_detect_with_permutation(context, observed_token, vocab_size, k, b, seed, model, tokenizer, n_runs=100, max_seed=100000, threshold=0.7):
#     """
#     SimHash Detection Logic with Permutation Test for p-value computation.
#     Determines whether a watermark exists in the observed token.

#     Inputs:
#     - context: Available tokens (text context).
#     - observed_token: The next token being analyzed for the watermark.
#     - vocab_size: Size of the vocabulary.
#     - k: Number of hash functions.
#     - b: Number of bits for the hash.
#     - seed: Seed for consistent random sampling.
#     - model: Pretrained language model.
#     - tokenizer: Tokenizer corresponding to the model.
#     - n_runs: Number of permutations for the null distribution.
#     - max_seed: Maximum seed value for generating permutations.
#     - threshold: P-value threshold for watermark detection.

#     Outputs:
#     - p_value: Computed p-value based on permutation test.
#     - result: "Watermark detected" or "No watermark detected" based on threshold.
#     """
#     # Step 1: Create SimHashWatermark instance
#     d = simple_encoder(context, model, tokenizer).size(-1)  # Determine embedding size dynamically
#     watermark = SimHashWatermark(d, vocab_size, k, b, seed)
    
#     # Print and compare sizes
#     # print(f"Embedding size (d): {d}")
#     # print(f"Watermark dimensionality: {watermark.d}")
#     # print(f"Vocabulary size: {watermark.vocab_size}")

#     # # Optional: Add an assertion to ensure sizes match expectations
#     # assert watermark.d == d, "Mismatch between embedding size and watermark dimensionality!"
#     # assert watermark.vocab_size == vocab_size, "Mismatch between vocabulary sizes!"

#     # Step 2: Embed context into vector v in R^d
#     embedded_context = simple_encoder(context, model, tokenizer)
#     assert embedded_context.size(-1) == d, "Embedding size must match Gaussian vector size!"

#     # Step 3: Compute observed test statistic
#     def compute_test_stat(token, null=False):
#         """
#         Helper function to compute the test statistic for a given token.
#         """
#         min_cost = float("inf")
#         for ell in range(k):
#             xi = watermark.sample_text_seed(embedded_context, ell)
#             xi_i = xi[token % xi.size(0)]  # Use token directly as index, ensure bounds
#             cost = -torch.log(1 - xi_i + 1e-9)
#             min_cost = min(min_cost, cost.item())
#         return min_cost

#     observed_result = compute_test_stat(observed_token)

#     # Step 4: Generate null distribution via permutations
#     generator = torch.Generator()
#     generator.manual_seed(seed)  # Set random seed for reproducibility
#     null_results = []

#     for _ in range(n_runs):
#         pi = torch.randperm(tokenizer.vocab_size, generator=generator)  # Generate random permutation
#         permuted_token = pi[observed_token]  # Apply permutation to observed token
#         null_results.append(compute_test_stat(permuted_token, null=True))

#     # Step 5: Compute p-value from null distribution
#     null_results = torch.tensor(sorted(null_results))
#     p_value = torch.searchsorted(null_results, observed_result, right=True).item() / len(null_results)

#     # Step 6: Analyze detection based on threshold
#     result = "Watermark detected" if p_value < threshold else "No watermark detected"

#     # Return p-value and result
#     return p_value, result, observed_result, null_results

def simhash_detect_with_permutation(context, observed_token, vocab_size, k, b, seed, model, tokenizer, n_runs=100, max_seed=100000, threshold=0.7):
    """
    SimHash Detection Logic with Permutation Test for p-value computation.
    Determines whether a watermark exists in the observed token.
    """
    # Step 1: Create SimHashWatermark instance
    d = simple_encoder(context, model, tokenizer).size(-1)  # Determine embedding size dynamically
    watermark = SimHashWatermark(d, vocab_size, k, b, seed)

    # Step 2: Embed context into vector v in R^d
    embedded_context = simple_encoder(context, model, tokenizer)
    assert embedded_context.size(-1) == d, "Embedding size must match Gaussian vector size!"

    # Step 3: Compute observed test statistic
    def compute_test_stat(token, null=False, token_label="Observed"):
        """
        Helper function to compute the test statistic for a given token.
        """
        min_cost = float("inf")
        for ell in range(k):
            xi = watermark.sample_text_seed(embedded_context, ell)
            xi_i = xi[token % xi.size(0)]  # Use token directly as index, ensure bounds
            cost = -torch.log(1 - xi_i + 1e-9)
            min_cost = min(min_cost, cost.item())

        # Debug print statements
        if token_label == "Observed":
            print(f"{token_label} Token={token}, xi[{token}]={xi[token]:.6f}, Cost={cost.item():.6f}")
        elif token_label == "Null":
            print(f"{token_label} Token={token}, xi[{token}]={xi[token]:.6f}, Cost={cost.item():.6f}")
        return min_cost

    observed_result = compute_test_stat(observed_token, token_label="Observed")

    # Step 4: Generate null distribution via permutations
    generator = torch.Generator()
    generator.manual_seed(seed)  # Set random seed for reproducibility
    null_results = []

    for _ in range(n_runs):
        pi = torch.randperm(tokenizer.vocab_size, generator=generator)  # Generate random permutation
        permuted_token = pi[observed_token]  # Apply permutation to observed token
        null_results.append(compute_test_stat(permuted_token, null=True, token_label="Null"))

    # Step 5: Compute p-value from null distribution
    null_results = torch.tensor(sorted(null_results))
    p_value = torch.searchsorted(null_results, observed_result, right=True).item() / len(null_results)
    # p_value = 1 - (count_below / len(null_results))
    p_value = 1 - p_value

    # Step 5: Compute p-value for P(null >= observed)
    # null_results = torch.tensor(sorted(null_results))
    # count_below = torch.searchsorted(null_results, observed_result, right=True).item()
    # p_value = 1 - (count_below / len(null_results))


    print(f"Null Distribution (first 10): {null_results[:10]}")
    print(f"Observed Result: {observed_result}")

    # Step 6: Analyze detection based on threshold
    result = "Watermark detected" if p_value < threshold else "No watermark detected"

    # Return p-value and result
    return p_value, result, observed_result, null_results


import matplotlib.pyplot as plt
import torch
from scipy.stats import skew, kurtosis

# Define helper analysis functions
def analyze_null_distribution(null_results, observed_result):
    """
    Visualize and summarize the null distribution and compare with the observed result.
    """
    null_results = torch.tensor(null_results)
    
    # Plot null distribution
    plt.figure(figsize=(10, 6))
    plt.hist(null_results, bins=30, alpha=0.7, label="Null Distribution", density=True, edgecolor='black')
    plt.axvline(x=observed_result, color='red', linestyle='--', label='Observed Result')
    plt.title("Null Distribution vs Observed Result")
    plt.xlabel("Test Statistic")
    plt.ylabel("Frequency (Normalized)")
    plt.legend()
    plt.show()
    
    # Compute and print metrics
    mean_null = torch.mean(null_results).item()
    var_null = torch.var(null_results).item()
    skew_null = skew(null_results.numpy())
    kurt_null = kurtosis(null_results.numpy())
    
    print(f"Null Distribution Metrics:")
    print(f"Mean: {mean_null:.4f}, Variance: {var_null:.4f}, Skewness: {skew_null:.4f}, Kurtosis: {kurt_null:.4f}")

def overlay_xi_null(null_results, xi_values):
    """
    Overlay observed xi distribution on the null distribution.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(null_results, bins=30, alpha=0.7, label="Null Distribution", density=True, edgecolor='black')
    plt.hist(xi_values, bins=30, alpha=0.7, label="Observed xi Distribution", density=True, edgecolor='black', color='green')
    plt.title("Comparison of Null Distribution and Observed xi")
    plt.xlabel("Values")
    plt.ylabel("Frequency (Normalized)")
    plt.legend()
    plt.show()

def simulate_known_watermark(null_results, simulated_observed):
    """
    Analyze the impact of a simulated known watermark.
    """
    null_results = torch.tensor(null_results)
    plt.figure(figsize=(10, 6))
    plt.hist(null_results, bins=30, alpha=0.7, label="Null Distribution", density=True, edgecolor='black')
    plt.axvline(x=simulated_observed, color='blue', linestyle='--', label='Simulated Watermark Result')
    plt.title("Null Distribution vs Simulated Watermark")
    plt.xlabel("Test Statistic")
    plt.ylabel("Frequency (Normalized)")
    plt.legend()
    plt.show()
    
    print(f"Simulated Watermark Observed Statistic: {simulated_observed:.4f}")

# Main block for sanity checks and analysis
if __name__ == "__main__":
    # Define test parameters
    vocab_size = 50265  # Replace with your model's vocabulary size
    k = 100
    b = 20
    seed = 42
    n_runs = 300
    threshold = 0.5

    # Load model and tokenizer
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Controlled test context and token
    test_context = "This is a controlled test sequence"
    test_observed_token = 42  # The token you forced in generation

    # Perform the detection sanity check and collect null results
    p_value, result, min_cost, null_results = simhash_detect_with_permutation(
        context=test_context,
        observed_token=test_observed_token,
        vocab_size=vocab_size,
        k=k,
        b=b,
        seed=seed,
        model=model,
        tokenizer=tokenizer,
        n_runs=n_runs,
        threshold=threshold
    )
    print(f"Sanity Check Detection Results: P-value={p_value}, Result={result}, Min Cost={min_cost}")

    # Analyze null distribution
    analyze_null_distribution(null_results, min_cost)

    # Overlay xi and null distribution (if applicable)
    xi_values = torch.ones(vocab_size) / vocab_size  # Example xi distribution
    overlay_xi_null(null_results, xi_values.numpy())

    # Simulate known watermark
    simulated_observed = min_cost + 0.1  # Example simulated statistic
    simulate_known_watermark(null_results, simulated_observed)
