import torch
from generate_simhash import SimHashWatermark

def simhash_detect_with_permutation(context, observed_token, encoder, d, k, b, seed, model, tokenizer, n_runs=100, max_seed=100000, threshold=0.5):
    """
    SimHash Detection Logic with Permutation Test for p-value computation.
    Determines whether a watermark exists in the observed token.

    Inputs:
    - context: Available tokens (text context).
    - observed_token: The next token being analyzed for the watermark.
    - encoder: Encoding function to embed the context into vector space.
    - d: Dimensionality of the embedding space.
    - k: Number of hash functions.
    - b: Number of bits for the hash.
    - seed: Seed for consistent random sampling.
    - model: Pretrained language model.
    - tokenizer: Tokenizer corresponding to the model.
    - test_stat: Function to compute the test statistic for watermark detection.
    - n_runs: Number of permutations for the null distribution.
    - max_seed: Maximum seed value for generating permutations.
    - threshold: P-value threshold for watermark detection.

    Outputs:
    - p_value: Computed p-value based on permutation test.
    - result: "Watermark detected" or "No watermark detected" based on threshold.
    """
    # Step 1: Create SimHashWatermark instance
    watermark = SimHashWatermark(d, k, b, seed)

    # Step 2: Embed context using encoder into vector v in R^d
    embedded_context = encoder(context, model, tokenizer)
    assert embedded_context.size(-1) == d, "Embedding size must match Gaussian vector size!"

    # Step 3: Compute observed test statistic
    def compute_test_stat(token, null=False):
        """
        Helper function to compute the test statistic for a given token.
        """
        min_cost = float("inf")
        for ell in range(k):
            xi = watermark.sample_text_seed(embedded_context, ell)
            token_index = token % d  # Ensure token index is within bounds
            xi_i = xi[token_index]
            cost = -torch.log(1 - xi_i + 1e-9)
            min_cost = min(min_cost, cost.item())
        return min_cost

    observed_result = compute_test_stat(observed_token)

    # Step 4: Generate null distribution via permutations
    generator = torch.Generator()
    generator.manual_seed(seed)  # Set random seed for reproducibility
    null_results = []

    for _ in range(n_runs):
        # Randomly permute tokens
        random_token = torch.randint(0, tokenizer.vocab_size, (1,), generator=generator).item()
        null_results.append(compute_test_stat(random_token, null=True))

    # Step 5: Compute p-value from null distribution
    null_results = sorted(null_results)
    p_value = sum(1 for null_result in null_results if null_result <= observed_result) / len(null_results)

    # Step 6: Analyze detection based on threshold
    result = "Watermark detected" if p_value < threshold else "No watermark detected"

    # Return p-value and result
    return p_value, result
