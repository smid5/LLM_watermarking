import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash, SimHashWatermark, simple_encoder
from detection_simhash import simhash_detect_with_permutation

# Advanced paraphrasing methods
def substitution_attack(tokens, p, vocab_size, distribution=None):
    if distribution is None:
        distribution = lambda x: torch.ones(size=(len(tokens), vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p * len(tokens))]
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs, 1).flatten()
    tokens[idx] = samples[idx]
    return tokens

def paraphrase_text(tokens, attack_type="substitution", intensity=0.3, vocab_size=None):
    if attack_type == "substitution":
        return substitution_attack(tokens.clone(), intensity, vocab_size)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

# Function to generate unrelated and unwatermarked text dynamically
def generate_unrelated_text(model, tokenizer, m, seed):
    torch.manual_seed(seed)
    prompts = tokenizer("", return_tensors="pt").input_ids  # Start with an empty prompt
    outputs = model.generate(prompts, max_length=m, num_return_sequences=1, do_sample=True, top_k=50, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main experiment function
def run_experiment():
    # Model and tokenizer setup
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Parameters
    k = 150   # Number of hash functions
    b = 30    # Bits per hash
    base_seed = 42  # Base seed for reproducibility
    vocab_size = tokenizer.vocab_size
    n = 256   # Sampling parameter (unused here)
    m = 50    # Number of tokens to generate
    num_samples = 1  # Increase number of samples for better analysis

    # Generate initial tokens (starting from an empty prompt)
    prompts = tokenizer("", return_tensors="pt").input_ids  # Empty context to start generation

    # Collect all xi values for each case
    watermarked_xis = []
    paraphrased_xis = []
    unrelated_xis = []

    # Helper function to collect xi values
    def collect_xi_values(text, seed, observed_token, text_type):
        d = simple_encoder(text, model, tokenizer).size(-1)
        watermark = SimHashWatermark(d, vocab_size, k, b, seed)
        embedded_context = simple_encoder(text, model, tokenizer)

        xis = []
        for ell in range(k):
            xi = watermark.sample_text_seed(embedded_context, ell)
            xis.extend(xi.tolist())  # Collect all xi values
        return xis

    for i in range(num_samples):
        # Generate a unique seed for each sample
        seed = base_seed + i

        # Generate watermarked text
        watermarked_tokens = generate_with_simhash(
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
        watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
        watermarked_xis.extend(
            collect_xi_values(watermarked_text, seed, watermarked_tokens[-1], "watermarked")
        )

        # Generate paraphrased text
        watermarked_tokens_tensor = torch.tensor(watermarked_tokens, dtype=torch.long)
        paraphrased_tokens = paraphrase_text(watermarked_tokens_tensor, attack_type="substitution", intensity=0.3, vocab_size=vocab_size)
        paraphrased_text = tokenizer.decode(paraphrased_tokens, skip_special_tokens=True)
        paraphrased_tokens = tokenizer(paraphrased_text, return_tensors="pt").input_ids[0]
        paraphrased_xis.extend(
            collect_xi_values(paraphrased_text, seed, paraphrased_tokens[-1], "paraphrased")
        )

        # Generate unrelated and unwatermarked text
        unrelated_text = generate_unrelated_text(model, tokenizer, m, seed)
        unrelated_tokens = tokenizer(unrelated_text, return_tensors="pt").input_ids[0]
        unrelated_xis.extend(
            collect_xi_values(unrelated_text, seed, unrelated_tokens[-1], "unrelated")
        )

    # Plot histograms of xi values separately
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.hist(watermarked_xis, bins=50, alpha=0.7, color="blue")
    plt.title("Watermarked Text xi Values")
    plt.xlabel("xi Values")
    plt.ylabel("Frequency")

    plt.subplot(3, 1, 2)
    plt.hist(paraphrased_xis, bins=50, alpha=0.7, color="yellow")
    plt.title("Paraphrased Text xi Values")
    plt.xlabel("xi Values")
    plt.ylabel("Frequency")

    plt.subplot(3, 1, 3)
    plt.hist(unrelated_xis, bins=50, alpha=0.7, color="red")
    plt.title("Unrelated Text xi Values")
    plt.xlabel("xi Values")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig("xi_histogram_output.png")

    plt.show()

if __name__ == "__main__":
    run_experiment()
