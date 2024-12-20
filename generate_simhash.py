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
    def __init__(self, d, k, b, seed):
        """
        Initialization: Precompute Gaussian vectors.
        Step 3: Use (seed, ell) to sample b Gaussian vectors r_1, ..., r_b in R^d.
        """
        self.d = d  # Dimensionality of the embedding space
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
        Step 6: Use text_seed to sample xi ~ Unif[(0, 1)]^d.
        - Deterministically generate xi using hash value as the seed.
        """
        hash_value = self.hash_function(vector, ell)  # Compute hash value
        generator = torch.Generator()
        generator.manual_seed(hash_value)  # Set random seed based on hash value
        return torch.rand(self.d, generator=generator)  # Generate uniform random vector xi

# Main generation function with SimHash watermarking
def generate_with_simhash(model, tokenizer, prompts, vocab_size, n, m, seeds, k, b):
    """
    Generation Algorithm: Combines SimHash with Exponential Minimum Sampling.
    Input:
    - context (prompts): Initial text input tokens.
    - LLM: Language model for next token prediction.
    - encoder: Embeds text context into vector space.
    - k, b: SimHash parameters.
    Output:
    - Generated text tokens.
    """
    # Step 1: Embed context into vector v in R^d
    context = tokenizer.decode(prompts[0], skip_special_tokens=True)
    embedded_context = simple_encoder(context, model, tokenizer)

    # Dynamically determine embedding dimensionality d
    d = embedded_context.size(-1)
    watermark = SimHashWatermark(d, k, b, seeds[0])  # Initialize SimHashWatermark

    inputs = prompts.to(model.device)  # Input tokens
    attn = torch.ones_like(inputs)  # Attention mask for the model
    past = None  # Cache past key values for efficiency
    temperature = 0.7  # Softmax temperature for sampling

    for i in range(m):  # Generate m tokens
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        logits = output.logits[:, -1]  # Logits for the next token

        # Step 7: Add Gumbel noise for stochastic sampling
        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(logits.shape).to(model.device)
        probs = torch.nn.functional.softmax((logits + gumbel_noise) / temperature, dim=-1)

        # Step 2: Sample ell uniformly from {1, ..., k}
        ell = torch.randint(0, k, (1,)).item()

        # Step 6: Generate xi ~ Unif[(0, 1)]^d using hash function hash_ell
        xi = watermark.sample_text_seed(embedded_context, ell)

        # Resize xi to match probs size
        xi_resized = torch.rand(probs.size(-1))  # Dynamically initialize xi_resized
        xi_resized[:min(xi.size(-1), probs.size(-1))] = xi[:min(xi.size(-1), probs.size(-1))]

        # Step 8: Exponential minimum sampling using log(xi) / probs
        scaled_probs = torch.log(xi_resized + 1e-9) / probs  # Logarithmic scaling
        next_token_id = torch.argmax(scaled_probs, dim=-1, keepdim=True)  # Select the next token

        # Append the next token to the input sequence
        inputs = torch.cat([inputs, next_token_id], dim=1)
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)  # Update attention mask

    return inputs[0].cpu().numpy().tolist()  # Convert generated tokens to a list for decoding
