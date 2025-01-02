import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def simple_encoder(text, model, tokenizer):
    """
    Encoder function: Converts input text into embeddings using the model's last hidden state.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1).squeeze()  # Mean pooling
    return embeddings

# SimHashWatermark class for watermarking logic
class SimHashWatermark:
    def __init__(self, d, vocab_size, k, b, seed):
        """
        Initialization: Precompute Gaussian vectors.
        Step 3: Use (seed, ell) to sample b Gaussian vectors r_1, ..., r_b in R^d.
        """
        self.d = d  # Dimensionality of the embedding space
        self.vocab_size = vocab_size  # Vocabulary size
        self.k = k  # Number of hash functions
        self.b = b  # Number of bits per hash
        self.seed = seed  # Seed for reproducibility

        torch.manual_seed(seed)  # Set random seed for reproducibility
        self.gaussian_vectors = [torch.randn(d) for _ in range(k * b)]  # Pre-generate Gaussian vectors

    def hash_function(self, vector, ell):
        """
        Step 4 & 5: Compute text_seed = hash_ell(sign(<r_1, v>), ..., sign(<r_b, v>)).
        - Select b Gaussian vectors for hash function ell.
        - Compute the sign of dot products <r_i, v>.
        - Convert binary values to an integer hash.
        """
        r_vectors = self.gaussian_vectors[ell * self.b:(ell + 1) * self.b]  # Select Gaussian vectors
        signs = [(torch.dot(r, vector.view(-1)) > 0).item() for r in r_vectors]  # Compute binary signs
        return int("".join(map(str, map(int, signs))), 2)  # Convert binary signs to an integer

    def sample_text_seed(self, vector, ell):
        """
        Step 6: Use text_seed to sample xi ~ Unif[(0,1)]^vocab_size.
        - Deterministically generate xi using hash value as the seed.
        """
        hash_value = self.hash_function(vector, ell)  # Compute hash value
        generator = torch.Generator()
        generator.manual_seed(hash_value)  # Set random seed based on hash value
        xi = torch.rand(self.vocab_size, generator=generator)  # Generate uniform random vector xi

        # Debugging: Check raw xi values and normalization
        print(f"[DEBUG] ell={ell}, hash_value={hash_value}, xi (raw first 10): {xi[:10]}")

        xi /= xi.sum()  # Normalize xi so that it sums to 1

        # Debugging: Verify normalization
        print(f"[DEBUG] ell={ell}, xi (normalized first 10): {xi[:10]}, sum={xi.sum().item()}")
    

        return xi


# def generate_with_simhash(model, tokenizer, prompts, vocab_size, n, m, seeds, k, b, random_offset=True):
#     """
#     Enhanced Generation Algorithm with SimHash and Exponential Minimum Sampling.
#     """
#     # Step 1: Embed context into vector v in R^d
#     context = tokenizer.decode(prompts[0], skip_special_tokens=True)
#     embedded_context = simple_encoder(context, model, tokenizer)

#     # Dynamically determine embedding dimensionality d
#     d = embedded_context.size(-1)
#     watermark = SimHashWatermark(d, vocab_size, k, b, seeds[0])  # Initialize SimHashWatermark

#     # Random offset for unpredictability
#     offset = torch.randint(n, size=(1,)) if random_offset else torch.zeros(1, dtype=torch.int64)

#     # Initialize inputs and attention mask
#     inputs = prompts.to(model.device)
#     attn = torch.ones_like(inputs)  # Attention mask
#     past = None  # For caching model's past key values
#     initial_temperature = 1.0  # Start with higher temperature
#     temperature_decay = 0.95  # Gradually reduce temperature

#     diversity_penalty = torch.ones(vocab_size, device=model.device)  # Initialize diversity penalty
#     token_history = torch.zeros(vocab_size, device=model.device)  # Track token frequencies

#     for i in range(m):  # Generate m tokens
#         with torch.no_grad():
#             if past:
#                 output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
#             else:
#                 output = model(inputs)

#         logits = output.logits[:, -1]

#         # Dynamically adjust temperature
#         temperature = max(0.7, initial_temperature * (temperature_decay ** i))
#         probs = torch.nn.functional.softmax(logits / temperature, dim=-1)

#         # Calculate entropy to adjust diversity penalty scaling
#         entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
#         entropy_factor = max(0.5, 1 - entropy / torch.log(torch.tensor(probs.size(-1), device=model.device)))

#         # Adjust diversity_penalty to match probs size
#         if diversity_penalty.size(0) != probs.size(-1):
#             diversity_penalty = torch.ones(probs.size(-1), device=model.device)  # Reinitialize
#             token_history = torch.zeros(probs.size(-1), device=model.device)  # Reinitialize token history

#         # Apply history-aware penalty to diversify tokens
#         history_penalty = torch.exp(-token_history / (i + 1))  # Decay based on token reuse
#         penalized_probs = (probs / diversity_penalty) * history_penalty
#         penalized_probs = penalized_probs / penalized_probs.sum()  # Renormalize

#         # Blend penalized and original probabilities
#         blended_probs = 0.7 * penalized_probs + 0.3 * probs

#         # Step 2: Sample ell uniformly from {1, ..., k}
#         ell = torch.randint(0, k, (1,)).item()

#         # Step 6: Use text_seed to sample xi
#         xi = watermark.sample_text_seed(embedded_context, ell)

#         # Ensure xi matches the size of probs
#         if xi.size(0) != blended_probs.size(-1):
#             xi = xi[:blended_probs.size(-1)]  # Trim xi if larger
#             xi = torch.nn.functional.pad(xi, (0, blended_probs.size(-1) - xi.size(0)))  # Pad xi if smaller

#         # Step 8: Exponential minimum sampling
#         alpha = 1.2  # Lower alpha for balanced watermark influence
#         scaled_probs = torch.log(xi + 1e-9) / (blended_probs ** alpha)

#         # Select the next token
#         next_token_id = torch.argmax(scaled_probs, dim=-1, keepdim=True)

#         # Update diversity penalty and token history
#         diversity_penalty[next_token_id] += 1  # Penalize selected token
#         token_history[next_token_id] += 1  # Track token usage

#         # Append the next token to the input sequence
#         inputs = torch.cat([inputs, next_token_id], dim=1)
#         attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)  # Update attention mask
#         past = output.past_key_values  # Update cached key values

#     return inputs[0].cpu().numpy().tolist()  # Convert generated tokens to a list for decoding

import matplotlib.pyplot as plt

def generate_with_simhash(model, tokenizer, prompts, vocab_size, n, m, seeds, k, b, random_offset=True):
    context = tokenizer.decode(prompts[0], skip_special_tokens=True)
    embedded_context = simple_encoder(context, model, tokenizer)

    d = embedded_context.size(-1)
    watermark = SimHashWatermark(d, vocab_size, k, b, seeds[0])

    offset = torch.randint(n, size=(1,)) if random_offset else torch.zeros(1, dtype=torch.int64)
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None

    diversity_penalty = torch.ones(vocab_size, device=model.device)
    token_history = torch.zeros(vocab_size, device=model.device)

    xi_values = []
    probabilities_log = []

    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        # Compute logits
        logits = output.logits[:, -1]

        # Sanity check: Override logits for the first few iterations
        # if i < 5:  # Apply to the first 5 iterations as a test
        #     logits = torch.zeros_like(logits)
        #     logits[0, 42] = 10.0  # Make token 42 highly probable

        # Normalize logits only if not overridden
        if i >= 5:
            logits = logits / logits.abs().max()

        # Compute probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        print(f"Iteration {i + 1}: Probabilities (first 10): {probs[:10].cpu().numpy()}")
        print(f"Iteration {i + 1}: Logits (first 10): {logits[:10].cpu().numpy()}")

        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        print(f"Iteration {i + 1}: Entropy: {entropy}")

        if diversity_penalty.size(0) != probs.size(-1):
            diversity_penalty = torch.ones(probs.size(-1), device=model.device)
            token_history = torch.zeros(probs.size(-1), device=model.device)

        history_penalty = torch.exp(-token_history / (i + 1))
        penalized_probs = (probs / diversity_penalty) * history_penalty
        penalized_probs = penalized_probs / penalized_probs.sum()

        blended_probs = 0.7 * penalized_probs + 0.3 * probs
        probabilities_log.append(blended_probs.cpu().numpy())

        ell = torch.randint(0, k, (1,)).item()
        xi = watermark.sample_text_seed(embedded_context, ell)
        xi_values.append(xi.cpu().numpy())

        # Ensure xi matches the size of probs
        if xi.size(0) != blended_probs.size(-1):
            xi = xi[:blended_probs.size(-1)]  # Trim xi if larger
            xi = torch.nn.functional.pad(xi, (0, blended_probs.size(-1) - xi.size(0)))  # Pad xi if smaller

        # Re-normalize xi after resizing
        xi /= xi.sum()

        alpha = 3.0
        scaled_probs = torch.log(xi + 1e-9) / (blended_probs ** alpha)
        next_token_id = torch.argmax(scaled_probs, dim=-1, keepdim=True)

        print(f"Probs Before xi Influence (iteration {i+1}): {probs[:10]}")
        print(f"Probs After xi Influence (iteration {i+1}): {scaled_probs[:10]}")


        diversity_penalty[next_token_id] += 1
        token_history[next_token_id] += 1

        inputs = torch.cat([inputs, next_token_id], dim=1)
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        past = output.past_key_values

    plot_debugging_info(xi_values, probabilities_log)
    return inputs[0].cpu().numpy().tolist()


def plot_debugging_info(xi_values, probabilities_log):
    """
    Plots debugging information for xi values and probabilities.

    Args:
        xi_values (list or torch.Tensor): Log-scaled xi values.
        probabilities_log (list or torch.Tensor): Log-scaled probabilities.
    """
    # Convert xi_values to a 1D NumPy array
    if isinstance(xi_values, list):
        xi_values = np.array(xi_values).flatten()
    else:
        xi_values = xi_values.flatten().cpu().numpy()

    # Convert probabilities_log to a 1D NumPy array
    if isinstance(probabilities_log, list):
        probabilities_log = np.array(probabilities_log).flatten()
    else:
        probabilities_log = probabilities_log.flatten().cpu().numpy()

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(xi_values, bins=50, alpha=0.7, label="xi Values", color="blue")
    plt.hist(probabilities_log, bins=50, alpha=0.7, label="Log Probabilities", color="green")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Debugging Information: xi Values vs. Log Probabilities")
    plt.legend()
    plt.grid(axis="y")
    plt.show()

    title = "Debugging Information: xi Values vs. Log Probabilities"
    plt.savefig(str(title) + ".png") 

def main():
    # Initialize model and tokenizer
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size

    # Set parameters
    seed = 42
    k = 5  # Number of hash functions
    b = 8  # Bits per hash
    n = 100  # Number of random offsets
    m = 20  # Number of tokens to generate
    prompts = tokenizer("Once upon a time,", return_tensors="pt").input_ids

    # Test simple_encoder
    context = tokenizer.decode(prompts[0], skip_special_tokens=True)
    embedded_context = simple_encoder(context, model, tokenizer)
    print(f"Context Embedding (size: {embedded_context.size()}): {embedded_context[:10]}")

    # Test SimHashWatermark
    d = embedded_context.size(-1)
    watermark = SimHashWatermark(d, vocab_size, k, b, seed)

    ell = 0  # Test with the first hash function
    xi = watermark.sample_text_seed(embedded_context, ell)
    print(f"Sampled xi (size: {xi.size()}): {xi[:10]}")

    # Test full generation with debugging plots
    generated_sequence = generate_with_simhash(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        vocab_size=vocab_size,
        n=n,
        m=m,
        seeds=[seed],
        k=k,
        b=b,
        random_offset=True,
    )

    # Decode and display the generated text
    generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")

if __name__ == "__main__":
    main()

