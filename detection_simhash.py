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

def simhash_detect_with_permutation(context, observed_sequence, vocab_size, k, b, model, tokenizer, seed=None, n_runs=100, max_seed=100000, threshold=0.1):
    """
    SimHash Detection Logic with Permutation Test for p-value computation.
    Determines whether a watermark exists in the observed sequence of tokens.
    """
    # Create SimHashWatermark instance
    d = simple_encoder(context, model, tokenizer).size(-1)
    watermark = SimHashWatermark(d, vocab_size, k, b, seed)

    # Embed context
    embedded_context = simple_encoder(context, model, tokenizer)
    assert embedded_context.size(-1) == d, "Embedding size must match Gaussian vector size!"

    # Function to compute test statistic for a token
    def compute_test_stat(token, null=False):
        min_cost = float("inf")
        for ell in range(k):
            xi = watermark.sample_text_seed(embedded_context, ell)
            xi_i = xi[token % xi.size(0)]
            cost = -torch.log(xi_i)
            min_cost = min(min_cost, cost.item())
        return min_cost

    # def compute_test_stat(token, null=False):
    #     total_cost = 0
    #     for ell in range(k):
    #         xi = watermark.sample_text_seed(embedded_context, ell)
    #         xi_i = xi[token % xi.size(0)]
    #         cost = -torch.log(xi_i + 1e-9)  # Adding a small constant to avoid log(0)
    #         total_cost += cost.item()
    #     average_cost = total_cost / k  # Calculate the average cost over all hash functions
    #     return average_cost

    # Compute observed test statistics for the sequence
    # observed_results = [compute_test_stat(token) for token in observed_sequence]
    observed_results = [(token, compute_test_stat(token)) for token in observed_sequence]
    # observed_average_cost = sum(observed_results) / len(observed_results)
    observed_average_cost = sum(cost for _, cost in observed_results) / len(observed_results)
    # individual_token_costs = {token: costs for token, (_, costs) in zip(observed_sequence, observed_results)}

    # Generate null distribution via permutations
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
    null_results = []

    for _ in range(n_runs):
        permuted_results = []
        pi = torch.randperm(tokenizer.vocab_size, generator=generator)
        for token in observed_sequence:
            permuted_token = pi[token]
            permuted_results.append(compute_test_stat(permuted_token, null=True))
        null_results.append(sum(permuted_results) / len(permuted_results))

    # Compute p-value from null distribution
    null_results = torch.tensor(sorted(null_results))
    p_value = torch.searchsorted(null_results, observed_average_cost, right=True).item() / len(null_results)

    # Analyze detection based on threshold
    result = "Watermark detected" if p_value < threshold else "No watermark detected"

    return p_value, result, observed_average_cost, observed_results
