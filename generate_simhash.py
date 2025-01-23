import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    def __init__(self, d, vocab_size, k, b, seed=None):
        """
        Initialization: Precompute Gaussian vectors.
        Step 3: Use (seed, ell) to sample b Gaussian vectors r_1, ..., r_b in R^d.
        """
        self.d = d  # Dimensionality of the embedding space
        self.vocab_size = vocab_size  # Vocabulary size
        self.k = k  # Number of hash functions
        self.b = b  # Number of bits per hash
        self.seed = seed  # Seed for reproducibility

        # torch.manual_seed(seed)  # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)  # Set random seed for reproducibility if seed is given
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
        Step 6: Use text_seed to sample xi ~ Unif[(0, 1)]^vocab_size.
        - Deterministically generate xi using hash value as the seed.
        """

        hash_value = self.hash_function(vector, ell)  # Compute hash value
        generator = torch.Generator()
        generator.manual_seed(hash_value)  # Set random seed based on hash value
        return torch.rand(self.vocab_size, generator=generator)  # Generate uniform random vector xi

def generate_with_simhash(model, tokenizer, prompts, vocab_size, n, m, k, b, seeds=None, random_offset=True):
    """
    Enhanced Generation Algorithm with SimHash and Exponential Minimum Sampling.
    """
    # Step 1: Embed context into vector v in R^d
    context = tokenizer.decode(prompts[0], skip_special_tokens=True)
    embedded_context = simple_encoder(context, model, tokenizer)

    # Dynamically determine embedding dimensionality d
    d = embedded_context.size(-1)
    watermark = SimHashWatermark(d, vocab_size, k, b, seeds)  # Initialize SimHashWatermark

    # Random offset for unpredictability
    offset = torch.randint(n, size=(1,)) if random_offset else torch.zeros(1, dtype=torch.int64)

    # Initialize inputs and attention mask
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)  # Attention mask
    past = None  # For caching model's past key values
    initial_temperature = 1.0  # Start with higher temperature
    temperature_decay = 0.95  # Gradually reduce temperature

    diversity_penalty = torch.ones(vocab_size, device=model.device)  # Initialize diversity penalty
    token_history = torch.zeros(vocab_size, device=model.device)  # Track token frequencies

    for i in range(m):  # Generate m tokens
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        logits = output.logits[:, -1]

        # Dynamically adjust temperature
        temperature = max(0.7, initial_temperature * (temperature_decay ** i))
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)

        # Calculate entropy to adjust diversity penalty scaling
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        entropy_factor = max(0.5, 1 - entropy / torch.log(torch.tensor(probs.size(-1), device=model.device)))

        # Adjust diversity_penalty to match probs size
        if diversity_penalty.size(0) != probs.size(-1):
            diversity_penalty = torch.ones(probs.size(-1), device=model.device)  # Reinitialize
            token_history = torch.zeros(probs.size(-1), device=model.device)  # Reinitialize token history

        # Apply history-aware penalty to diversify tokens
        history_penalty = torch.exp(-token_history / (i + 1))  # Decay based on token reuse
        penalized_probs = (probs / diversity_penalty) * history_penalty
        penalized_probs = penalized_probs / penalized_probs.sum()  # Renormalize

        # Blend penalized and original probabilities
        blended_probs = 0.7 * penalized_probs + 0.3 * probs

        # Step 2: Sample ell uniformly from {1, ..., k}
        ell = torch.randint(0, k, (1,)).item()

        # Step 6: Use text_seed to sample xi
        xi = watermark.sample_text_seed(embedded_context, ell)

        # Ensure xi matches the size of probs
        if xi.size(0) != blended_probs.size(-1):
            xi = xi[:blended_probs.size(-1)]  # Trim xi if larger
            xi = torch.nn.functional.pad(xi, (0, blended_probs.size(-1) - xi.size(0)))  # Pad xi if smaller

        # Step 8: Exponential minimum sampling
        alpha = 1.2  # Lower alpha for balanced watermark influence
        scaled_probs = torch.log(xi + 1e-9) / (blended_probs ** alpha)

        # Select the next token
        next_token_id = torch.argmax(scaled_probs, dim=-1, keepdim=True)

        # Update diversity penalty and token history
        diversity_penalty[next_token_id] += 1  # Penalize selected token
        token_history[next_token_id] += 1  # Track token usage

        # Append the next token to the input sequence
        inputs = torch.cat([inputs, next_token_id], dim=1)
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)  # Update attention mask
        past = output.past_key_values  # Update cached key values

    return inputs[0].cpu().numpy().tolist()  # Convert generated tokens to a list for decoding