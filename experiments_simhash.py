import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash, SimHashWatermark
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

def deletion_attack(tokens, p):
    idx = torch.randperm(len(tokens))[:int(p * len(tokens))]
    keep = torch.ones(len(tokens), dtype=torch.bool)
    keep[idx] = False
    tokens = tokens[keep]
    return tokens

def insertion_attack(tokens, p, vocab_size, distribution=None):
    if distribution is None:
        distribution = lambda x: torch.ones(size=(len(tokens), vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p * len(tokens))]
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs, 1)
    for i in idx.sort(descending=True).values:
        tokens = torch.cat([tokens[:i], samples[i], tokens[i:]])
    return tokens

def paraphrase_text(tokens, attack_type="substitution", intensity=0.1, vocab_size=None):
    if attack_type == "substitution":
        return substitution_attack(tokens.clone(), intensity, vocab_size)
    elif attack_type == "deletion":
        return deletion_attack(tokens.clone(), intensity)
    elif attack_type == "insertion":
        return insertion_attack(tokens.clone(), intensity, vocab_size)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

# Function to generate unrelated and unwatermarked text dynamically
# def generate_unrelated_text(model, tokenizer, m, seed=None):
#     torch.manual_seed(seed)
#     prompts = tokenizer("", return_tensors="pt").input_ids  # Start with an empty prompt
#     outputs = model.generate(prompts, max_length=m, num_return_sequences=1, do_sample=True, top_k=50, temperature=1.5)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)
# import torch

def generate_unrelated_text(model, tokenizer, m, seed=None):
    """
    Generate text that is not related to a specific input prompt.
    
    Args:
    - model (transformers.PreTrainedModel): The language model to use for text generation.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer corresponding to the model.
    - m (int): The maximum length of the text to generate.
    - seed (int, optional): The seed for reproducibility. If None, no seed is set.

    Returns:
    - str: The generated text.
    """
    # Set the seed only if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Generate text
    prompts = tokenizer("", return_tensors="pt").input_ids  # Start with an empty prompt
    outputs = model.generate(
        prompts, 
        max_length=m, 
        num_return_sequences=1, 
        do_sample=True, 
        top_k=50, 
        temperature=1.5
    )

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to compare xi value distributions
def plot_xi_distributions(watermarked_xis, paraphrased_xis, unrelated_xis):
    plt.hist(watermarked_xis, bins=50, alpha=0.7, label="Watermarked", color="blue")
    plt.hist(paraphrased_xis, bins=50, alpha=0.7, label="Paraphrased", color="red")
    plt.hist(unrelated_xis, bins=50, alpha=0.7, label="Unrelated", color="yellow")
    plt.legend()
    plt.title("Distribution of xi Values")
    plt.xlabel("xi Value")
    plt.ylabel("Frequency")

    plt.savefig("Image_outputs/Distribution of xi Values.png")
    plt.show()

# Function to analyze substitution attack's effect on token probabilities
def analyze_substitution_effect(original_tokens, paraphrased_tokens, model, tokenizer):
    replaced_indices = (original_tokens != paraphrased_tokens).nonzero(as_tuple=True)[0]
    original_probs = []
    new_probs = []

    for idx in replaced_indices:
        with torch.no_grad():
            logits = model(original_tokens.unsqueeze(0)).logits
            probs = torch.softmax(logits[0, idx], dim=-1)
            original_probs.append(probs[original_tokens[idx]].item())
            new_probs.append(probs[paraphrased_tokens[idx]].item())

    plt.scatter(original_probs, new_probs, alpha=0.7)
    plt.xlabel("Original Token Probability")
    plt.ylabel("Replaced Token Probability")
    plt.title("Effect of Substitution on Token Probabilities")

    plt.savefig("Image_outputs/Effect of Substitution on Token Probabilities")
    plt.show()

# Function to analyze and plot token probability distributions
def plot_token_probabilities(tokens, model, tokenizer, label):
    """
    Plots the token probability distribution for a given set of tokens.
    
    Args:
        tokens (list): List of tokens to analyze.
        model: Pretrained language model.
        tokenizer: Tokenizer associated with the language model.
        label (str): Label for the plot (e.g., "Watermarked", "Paraphrased").
    """
    probabilities = []
    for token in tokens:
        with torch.no_grad():
            logits = model(torch.tensor(tokens).unsqueeze(0)).logits
            probs = torch.softmax(logits[0, -1], dim=-1)  # Probabilities for the last token
            probabilities.append(probs[token].item())  # Extract probability for the given token

    plt.figure(figsize=(10, 6))
    plt.hist(probabilities, bins=50, alpha=0.7, label=label, color="blue" if label == "Watermarked" else "green" if label == "Paraphrased" else "red")
    title = f"Token Probability Distribution: {label}"
    plt.title(f"Token Probability Distribution: {label}")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y")

    plt.savefig("Image_outputs/"+ str(title) + ".png")

    plt.show()

def plot_detailed_results(detailed_results, tokenizer, title="Token Cost Distribution"):
    tokens, costs = zip(*detailed_results)  # Unpack the list of tuples into two separate tuples
    words = [tokenizer.decode([token]) for token in tokens]  # Convert token IDs to words
    
    plt.figure(figsize=(12, 8))
    plt.bar(words, costs, color='blue')  # Plot using words as labels
    plt.xlabel('Tokens')
    plt.ylabel('Minimum Costs')
    plt.title(title)
    plt.xticks(rotation=90)  # Rotate labels to fit longer words
    plt.tight_layout()  # Adjust layout to make room for x-axis labels
    plt.savefig("Image_outputs/" + str(title))
    plt.show()

# Updated main experiment function
def run_experiment():
    # Model and tokenizer setup
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Parameters
    k = 30   # Number of hash functions
    b = 10    # Bits per hash
    # base_seed = 42  # Base seed for reproducibility
    vocab_size = tokenizer.vocab_size
    n = 256   # Sampling parameter (unused here)
    m = 50    # Number of tokens to generate
    num_samples = 10  # Increase number of samples for better analysis

    # Generate initial tokens (starting from an empty prompt)
    prompts = tokenizer("", return_tensors="pt").input_ids  # Empty context to start generation

    # Collect min_costs for each case
    watermarked_costs = []
    paraphrased_costs = []
    unrelated_costs = []

    # Helper lists for xi analysis
    watermarked_xis = []
    paraphrased_xis = []
    unrelated_xis = []

    # Open a file to write outputs
    with open("text_outputs.txt", "w") as output_file:
        for i in range(num_samples):
            # Generate a unique seed for each sample
            # seed = base_seed + i

            # Generate watermarked text
            watermarked_tokens = generate_with_simhash(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                vocab_size=vocab_size,
                n=n,
                m=m,
                # seeds=[seed],
                k=k,
                b=b,
            )
            watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

            # Collect xi values for watermarked text
            watermarked_xis.extend(watermarked_tokens)

            wm_p_value, _, wm_min_cost, wm_ind_costs = simhash_detect_with_permutation(
                context=watermarked_text,  # Use the whole generated text as context
                observed_sequence=watermarked_tokens,  # Use the whole sequence of tokens
                vocab_size=vocab_size,
                k=k,
                b=b,
                # seed=seed,
                model=model,
                tokenizer=tokenizer
            )
            watermarked_costs.append(wm_min_cost)

            # Plot token probabilities for watermarked text
            plot_token_probabilities(watermarked_tokens, model, tokenizer, "Watermarked")
            # plot_token_costs(watermarked_tokens, model, tokenizer, watermarked_text, k, b, seed, "Watermarked")
            plot_detailed_results(wm_ind_costs, tokenizer, title=f"Token Cost Distribution Watermarked for Sample {i+1}")

            # Generate paraphrased text
            watermarked_tokens_tensor = torch.tensor(watermarked_tokens, dtype=torch.long)
            paraphrased_tokens = paraphrase_text(watermarked_tokens_tensor, attack_type="substitution", intensity=0.3, vocab_size=vocab_size)
            paraphrased_text = tokenizer.decode(paraphrased_tokens, skip_special_tokens=True)

            # Collect xi values for paraphrased text
            paraphrased_xis.extend(paraphrased_tokens)

            para_p_value, _, para_min_cost, para_ind_costs = simhash_detect_with_permutation(
                context=paraphrased_text,  # Use the whole generated text as context
                observed_sequence=paraphrased_tokens,  # Use the whole sequence of tokens
                vocab_size=vocab_size,
                k=k,
                b=b,
                # seed=seed,
                model=model,
                tokenizer=tokenizer
            )
            paraphrased_costs.append(para_min_cost)

            # Plot token probabilities for paraphrased text
            plot_token_probabilities(paraphrased_tokens, model, tokenizer, "Paraphrased")
            plot_detailed_results(para_ind_costs, tokenizer, title=f"Token Cost Distribution Paraphrased for Sample {i+1}")
            # plot_token_costs(paraphrased_tokens, model, tokenizer, paraphrased_text, k, b, seed, "Paraphrased")

            # Generate unrelated and unwatermarked text
            unrelated_text = generate_unrelated_text(model, tokenizer, m)
            unrelated_tokens = tokenizer(unrelated_text, return_tensors="pt").input_ids[0]

            # Collect xi values for unrelated text
            unrelated_xis.extend(unrelated_tokens)

            unrelated_p_value, _, unrelated_min_cost, unrelated_ind_costs = simhash_detect_with_permutation(
                context=unrelated_text,  # Use the whole generated text as context
                observed_sequence=unrelated_tokens,  # Use the whole sequence of tokens
                vocab_size=vocab_size,
                k=k,
                b=b,
                # seed=seed,
                model=model,
                tokenizer=tokenizer
            )
            unrelated_costs.append(unrelated_min_cost)

            # Plot token probabilities for unrelated text
            plot_token_probabilities(unrelated_tokens, model, tokenizer, "Unrelated")
            plot_detailed_results(unrelated_ind_costs, tokenizer, title=f"Token Cost Distribution Unrelated for Sample {i+1}")
            # plot_token_costs(unrelated_tokens, model, tokenizer, unrelated_text, k, b, seed, "Unrelated")

            # Print and write the generated texts
            output = (
                f"ITERATION {i + 1}\n"
                f"Watermarked text: {watermarked_text}\n"
                f"Paraphrased text: {paraphrased_text}\n"
                f"Unrelated text: {unrelated_text}\n"
                f"Watermarked Min Cost: {wm_min_cost:.4f}\n"
                f"Paraphrased Min Cost: {para_min_cost:.4f}\n"
                f"Unrelated Min Cost: {unrelated_min_cost:.4f}\n\n"
            )
            print(output)
            output_file.write(output)

    # Compute average min_costs
    avg_watermarked = np.mean(watermarked_costs)
    avg_paraphrased = np.mean(paraphrased_costs)
    avg_unrelated = np.mean(unrelated_costs)

    # Print the averages
    print("Average Min Cost Values:")
    print(f"Watermarked Text: {avg_watermarked:.4f}")
    print(f"Attack Text: {avg_paraphrased:.4f}")
    print(f"Unrelated Text: {avg_unrelated:.4f}\n")

    # Write the averages to the file
    with open("text_outputs.txt", "a") as output_file:
        output_file.write("Average Min Cost Values:\n")
        output_file.write(f"Watermarked Text: {avg_watermarked:.4f}\n")
        output_file.write(f"Paraphrased Text: {avg_paraphrased:.4f}\n")
        output_file.write(f"Unrelated Text: {avg_unrelated:.4f}\n\n")

    # Plot xi distributions
    plot_xi_distributions(watermarked_xis, paraphrased_xis, unrelated_xis)

    analyze_substitution_effect(torch.tensor(watermarked_tokens, dtype=torch.long), paraphrased_tokens, model, tokenizer)

    # Plot bar chart of average min_costs
    plt.figure(figsize=(10, 6))
    categories = ["Watermarked Text", "Paraphrased Text", "Unrelated Text"]
    averages = [avg_watermarked, avg_paraphrased, avg_unrelated]

    plt.bar(categories, averages, color=["blue", "green", "red"])
    plt.xlabel("Text Type")
    plt.ylabel("Average cost")
    plt.title("Average Min Cost for Different Text Types")

    plt.savefig("Image_outputs/Average Min Cost for Different Text Types.png")

    plt.grid(axis="y")
    plt.show()

if __name__ == "__main__":
    run_experiment()

