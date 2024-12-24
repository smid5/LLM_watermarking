import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import SimHashWatermark

def simple_encoder(text, model, tokenizer):
    """
    Encoder function: Converts input text into embeddings using the model's last hidden state.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze()  # Mean pooling
    return embeddings

def simhash_detect_with_permutation(context, observed_token, vocab_size, k, b, seed, model, tokenizer, n_runs=100, max_seed=100000, threshold=0.1):
    """
    SimHash Detection Logic with Permutation Test for p-value computation.
    Determines whether a watermark exists in the observed token.

    Inputs:
    - context: Available tokens (text context).
    - observed_token: The next token being analyzed for the watermark.
    - vocab_size: Size of the vocabulary.
    - k: Number of hash functions.
    - b: Number of bits for the hash.
    - seed: Seed for consistent random sampling.
    - model: Pretrained language model.
    - tokenizer: Tokenizer corresponding to the model.
    - n_runs: Number of permutations for the null distribution.
    - max_seed: Maximum seed value for generating permutations.
    - threshold: P-value threshold for watermark detection.

    Outputs:
    - p_value: Computed p-value based on permutation test.
    - result: "Watermark detected" or "No watermark detected" based on threshold.
    """
    # Step 1: Create SimHashWatermark instance
    d = simple_encoder(context, model, tokenizer).size(-1)  # Determine embedding size dynamically
    watermark = SimHashWatermark(d, vocab_size, k, b, seed)
    
    # Print and compare sizes
    # print(f"Embedding size (d): {d}")
    # print(f"Watermark dimensionality: {watermark.d}")
    # print(f"Vocabulary size: {watermark.vocab_size}")

    # # Optional: Add an assertion to ensure sizes match expectations
    # assert watermark.d == d, "Mismatch between embedding size and watermark dimensionality!"
    # assert watermark.vocab_size == vocab_size, "Mismatch between vocabulary sizes!"

    # Step 2: Embed context into vector v in R^d
    embedded_context = simple_encoder(context, model, tokenizer)
    assert embedded_context.size(-1) == d, "Embedding size must match Gaussian vector size!"

    # Step 3: Compute observed test statistic
    def compute_test_stat(token, null=False):
        """
        Helper function to compute the test statistic for a given token.
        """
        min_cost = float("inf")
        for ell in range(k):
            xi = watermark.sample_text_seed(embedded_context, ell)
            xi_i = xi[token % xi.size(0)]  # Use token directly as index, ensure bounds
            cost = -torch.log(1 - xi_i + 1e-9)
            min_cost = min(min_cost, cost.item())
        return min_cost

    observed_result = compute_test_stat(observed_token)

    # Step 4: Generate null distribution via permutations
    generator = torch.Generator()
    generator.manual_seed(seed)  # Set random seed for reproducibility
    null_results = []

    for _ in range(n_runs):
        pi = torch.randperm(tokenizer.vocab_size, generator=generator)  # Generate random permutation
        permuted_token = pi[observed_token]  # Apply permutation to observed token
        null_results.append(compute_test_stat(permuted_token, null=True))

    # Step 5: Compute p-value from null distribution
    null_results = torch.tensor(sorted(null_results))
    p_value = torch.searchsorted(null_results, observed_result, right=True).item() / len(null_results)

    # Step 6: Analyze detection based on threshold
    result = "Watermark detected" if p_value < threshold else "No watermark detected"

    # Return p-value and result
    return p_value, result, observed_result
