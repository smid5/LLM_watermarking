import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def simple_encoder(text, model, tokenizer):
    """
    Step 1: Embed context into vector v in R^d.
    Converts input text into embeddings using the model's last hidden state.
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

def sample_and_scale_probs(embedded_context, probs, watermark, k):
    """
    Sample xi using SimHash and modify the token probabilities.
    """
    # Step 2: Sample ell uniformly from {1, ..., k}
    ell = torch.randint(0, k, (1,)).item()

    # Step 6: Use text_seed to sample xi ~ Unif[(0,1)]^vocab size
    xi = watermark.sample_text_seed(embedded_context, ell)
    
    # Ensure xi is properly aligned with the size of probs
    if xi.size(0) != probs.size(-1):
        if xi.size(0) > probs.size(-1):
            xi = xi[:probs.size(-1)]  # Trim xi if larger
        else:
            xi = torch.nn.functional.pad(xi, (0, probs.size(-1) - xi.size(0)))  # Pad xi if smaller
    
    # Normalize xi to use as probability adjustments
    xi_probs = xi / xi.sum()

    # Step 8: Apply exponential minimum sampling to get i* = max_j log(xi_j) / p_j
    scaled_probs = torch.pow(probs, 1.2) * xi_probs  # Apply the watermark with stronger influence
    return scaled_probs

def generate_with_simhash(model, tokenizer, prompts, vocab_size, n, m, k, b, seeds=None, random_offset=True):
    """
    Enhanced Generation Algorithm with SimHash and stronger watermarking influence.
    """
    # Step 1: Embed context using encoder into vector v in R^d
    context = tokenizer.decode(prompts[0], skip_special_tokens=True)
    embedded_context = simple_encoder(context, model, tokenizer)

    # Instantiate SimHashWatermark with the parameters including seeds
    watermark = SimHashWatermark(embedded_context.size(-1), vocab_size, k, b, seeds)

    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    temperature = 1.0

    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        # Step 7: Evaluate probability distribution p = LLM(context)
        probs = torch.nn.functional.softmax(output.logits[:, -1] / temperature, dim=-1)
        scaled_probs = sample_and_scale_probs(embedded_context, probs, watermark, k)

        # Select the next token based on the scaled probabilities
        next_token_id = torch.argmax(scaled_probs, dim=-1, keepdim=True)

        # Append the next token to the input sequence for the next iteration
        inputs = torch.cat([inputs, next_token_id], dim=1)
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        past = output.past_key_values

    return inputs[0].cpu().numpy().tolist()
