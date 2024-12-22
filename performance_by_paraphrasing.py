# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from generate_simhash import generate_with_simhash
# from detection_simhash import simhash_detect_with_permutation


# # Function to generate controlled paraphrased text
# def controlled_paraphrase(text, intensity=0.5):
#     """
#     Paraphrases text with a controllable intensity.
#     - Intensity: Fraction of words to paraphrase (0 to 1).
#     """
#     words = text.split()
#     num_changes = int(len(words) * intensity)
#     for _ in range(num_changes):
#         idx = torch.randint(0, len(words), (1,)).item()
#         words[idx] = words[idx][::-1]  # Example transformation: reverse the word
#     return " ".join(words)


# # Function to generate unrelated text
# def generate_unrelated_text(model, tokenizer, seed):
#     torch.manual_seed(seed)
#     random_topics = [
#         "astronomy", "cooking", "robotics", "music theory",
#         "ancient history", "quantum mechanics", "gardening", "sports analysis"
#     ]
#     random_topic = random_topics[seed % len(random_topics)]
#     random_prompt = f"Write a brief introduction to {random_topic}."
#     prompts = tokenizer(random_prompt, return_tensors="pt").input_ids
#     outputs = model.generate(prompts, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, temperature=0.7)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# # Function to evaluate detection by paraphrasing intensity
# def evaluate_by_paraphrasing_intensity(model, tokenizer, context, prompts, d, k, b, n, m, seed, num_samples, intensities):
#     performance_by_intensity = []

#     for intensity in intensities:
#         tp = 0  # True positives
#         total = 0

#         for sample in range(num_samples):
#             sample_seed = seed + sample

#             # Generate watermarked text
#             watermarked_tokens = generate_with_simhash(
#                 model=model,
#                 tokenizer=tokenizer,
#                 prompts=prompts,
#                 vocab_size=tokenizer.vocab_size,
#                 n=n,
#                 m=m,
#                 seeds=[sample_seed],
#                 k=k,
#                 b=b,
#             )
#             watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

#             # Paraphrase with the current intensity
#             paraphrased_text = controlled_paraphrase(watermarked_text, intensity=intensity)
#             paraphrased_tokens = tokenizer(paraphrased_text, return_tensors="pt").input_ids[0]

#             # Detect watermark
#             para_p_value, para_result, _ = simhash_detect_with_permutation(
#                 context=context,
#                 observed_token=int(paraphrased_tokens[-1]),
#                 d=d,
#                 k=k,
#                 b=b,
#                 seed=sample_seed,
#                 model=model,
#                 tokenizer=tokenizer,
#                 n_runs=100,
#                 threshold=0.5,
#             )
#             if para_result == "Watermark detected":
#                 tp += 1

#             total += 1

#         # Calculate detection rate for this intensity
#         detection_rate = tp / total
#         performance_by_intensity.append(detection_rate)

#     return performance_by_intensity


# # Plot performance by paraphrasing intensity
# def plot_paraphrasing_performance(intensities, performance):
#     plt.figure(figsize=(10, 6))
#     plt.plot(intensities, performance, marker="o", label="Detection Rate")
#     plt.xlabel("Paraphrasing Intensity")
#     plt.ylabel("Detection Rate (TPR)")
#     plt.title("Performance by Paraphrasing Intensity")
#     plt.grid()
#     plt.legend()
#     plt.show()


# # Main function to run experiments
# def run_experiment():
#     # Model and tokenizer setup
#     model_name = "facebook/opt-1.3b"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)

#     # Parameters
#     d = 2048
#     n = 256
#     m = 50
#     base_seed = 42
#     num_samples = 5

#     # k and b values to test
#     k = 100  # Fixed value for k
#     b = 30   # Fixed value for b

#     # Input context
#     context = "Once upon a time, in a faraway land, a boy's heart's aching."
#     prompts = tokenizer(context, return_tensors="pt").input_ids

#     # Paraphrasing intensities to evaluate
#     paraphrasing_intensities = [0.1, 0.3, 0.5, 0.7, 0.9]

#     # Evaluate detection performance by paraphrasing intensity
#     performance = evaluate_by_paraphrasing_intensity(
#         model, tokenizer, context, prompts, d, k, b, n, m, base_seed, num_samples, paraphrasing_intensities
#     )

#     # Plot performance
#     plot_paraphrasing_performance(paraphrasing_intensities, performance)


# if __name__ == "__main__":
#     run_experiment()

# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from generate_simhash import generate_with_simhash
# from detection_simhash import simhash_detect_with_permutation

# # Function to generate controlled paraphrased text
# def controlled_paraphrase(text, intensity=0.5):
#     """
#     Paraphrases text with a controllable intensity.
#     - Intensity: Fraction of words to paraphrase (0 to 1).
#     """
#     words = text.split()
#     num_changes = int(len(words) * intensity)
#     for _ in range(num_changes):
#         idx = torch.randint(0, len(words), (1,)).item()
#         words[idx] = words[idx][::-1]  # Example transformation: reverse the word
#     return " ".join(words)

# # Function to generate unrelated text
# def generate_unrelated_text(model, tokenizer, seed):
#     torch.manual_seed(seed)
#     random_topics = [
#         "astronomy", "cooking", "robotics", "music theory",
#         "ancient history", "quantum mechanics", "gardening", "sports analysis"
#     ]
#     random_topic = random_topics[seed % len(random_topics)]
#     random_prompt = f"Write a brief introduction to {random_topic}."
#     prompts = tokenizer(random_prompt, return_tensors="pt").input_ids
#     outputs = model.generate(prompts, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, temperature=0.7)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Function to evaluate detection by paraphrasing intensity
# def evaluate_by_paraphrasing_intensity(model, tokenizer, d, k, b, n, m, seed, num_samples, intensities):
#     performance_by_intensity = []

#     for intensity in intensities:
#         tp = 0  # True positives
#         total = 0

#         for sample in range(num_samples):
#             sample_seed = seed + sample

#             # Generate watermarked text
#             prompts = tokenizer("", return_tensors="pt").input_ids  # Start with an empty context
#             watermarked_tokens = generate_with_simhash(
#                 model=model,
#                 tokenizer=tokenizer,
#                 prompts=prompts,
#                 vocab_size=tokenizer.vocab_size,
#                 n=n,
#                 m=m,
#                 seeds=[sample_seed],
#                 k=k,
#                 b=b,
#             )
#             watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)

#             # Paraphrase with the current intensity
#             paraphrased_text = controlled_paraphrase(watermarked_text, intensity=intensity)
#             paraphrased_tokens = tokenizer(paraphrased_text, return_tensors="pt").input_ids[0]

#             # Detect watermark
#             para_p_value, para_result, _ = simhash_detect_with_permutation(
#                 context=tokenizer.decode(watermarked_tokens[:-1], skip_special_tokens=True),
#                 observed_token=int(paraphrased_tokens[-1]),
#                 d=d,
#                 k=k,
#                 b=b,
#                 seed=sample_seed,
#                 model=model,
#                 tokenizer=tokenizer,
#                 n_runs=100,
#                 threshold=0.5,
#             )
#             if para_result == "Watermark detected":
#                 tp += 1

#             total += 1

#         # Calculate detection rate for this intensity
#         detection_rate = tp / total
#         performance_by_intensity.append(detection_rate)

#     return performance_by_intensity

# # Plot performance by paraphrasing intensity
# def plot_paraphrasing_performance(intensities, performance):
#     plt.figure(figsize=(10, 6))
#     plt.plot(intensities, performance, marker="o", label="Detection Rate")
#     plt.xlabel("Paraphrasing Intensity")
#     plt.ylabel("Detection Rate (TPR)")
#     plt.title("Performance by Paraphrasing Intensity")
#     plt.grid()
#     plt.legend()
#     plt.show()

# # Main function to run experiments
# def run_experiment():
#     # Model and tokenizer setup
#     model_name = "facebook/opt-1.3b"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)

#     # Parameters
#     d = 2048
#     n = 256
#     m = 50
#     base_seed = 42
#     num_samples = 5

#     # k and b values to test
#     k = 100  # Fixed value for k
#     b = 30   # Fixed value for b

#     # Paraphrasing intensities to evaluate
#     paraphrasing_intensities = [0.1, 0.3, 0.5, 0.7, 0.9]

#     # Evaluate detection performance by paraphrasing intensity
#     performance = evaluate_by_paraphrasing_intensity(
#         model, tokenizer, d, k, b, n, m, base_seed, num_samples, paraphrasing_intensities
#     )

#     # Plot performance
#     plot_paraphrasing_performance(paraphrasing_intensities, performance)

# if __name__ == "__main__":
#     run_experiment()

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash
from detection_simhash import simhash_detect_with_permutation

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

def evaluate_by_paraphrasing_intensity(
    model, tokenizer, d, k, b, n, m, seed, num_samples, intensities, attack_type="substitution"
):
    performance_by_intensity = []

    for intensity in intensities:
        tp = 0  # True positives
        total = 0

        for sample in range(num_samples):
            sample_seed = seed + sample

            # Generate watermarked text
            prompts = tokenizer("", return_tensors="pt").input_ids  # Start with an empty context
            watermarked_tokens = generate_with_simhash(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                vocab_size=tokenizer.vocab_size,
                n=n,
                m=m,
                seeds=[sample_seed],
                k=k,
                b=b,
            )
            watermarked_tokens_tensor = torch.tensor(watermarked_tokens, dtype=torch.long)  # Convert to tensor

            # Apply attack based on the specified type
            if attack_type == "substitution":
                paraphrased_tokens = substitution_attack(
                    watermarked_tokens_tensor.clone(), intensity, vocab_size=tokenizer.vocab_size
                )
            elif attack_type == "deletion":
                paraphrased_tokens = deletion_attack(watermarked_tokens_tensor.clone(), intensity)
            elif attack_type == "insertion":
                paraphrased_tokens = insertion_attack(
                    watermarked_tokens_tensor.clone(), intensity, vocab_size=tokenizer.vocab_size
                )
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            # Detect watermark
            paraphrased_text = tokenizer.decode(paraphrased_tokens, skip_special_tokens=True)
            paraphrased_tokens_tensor = tokenizer(paraphrased_text, return_tensors="pt").input_ids[0]
            para_p_value, para_result, _ = simhash_detect_with_permutation(
                context=tokenizer.decode(watermarked_tokens[:-1], skip_special_tokens=True),
                observed_token=int(paraphrased_tokens_tensor[-1]),
                d=d,
                k=k,
                b=b,
                seed=sample_seed,
                model=model,
                tokenizer=tokenizer,
                n_runs=100,
                threshold=0.5,
            )
            if para_result == "Watermark detected":
                tp += 1

            total += 1

        # Calculate detection rate for this intensity
        detection_rate = tp / total
        performance_by_intensity.append(detection_rate)

    return performance_by_intensity

def plot_paraphrasing_performance(intensities, performance):
    plt.figure(figsize=(10, 6))
    plt.plot(intensities, performance, marker="o", label="Detection Rate")
    plt.xlabel("Paraphrasing Intensity")
    plt.ylabel("Detection Rate (TPR)")
    plt.title("Performance by Paraphrasing Intensity")
    plt.grid()
    plt.legend()
    plt.show()

def run_experiment():
    # Model and tokenizer setup
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Parameters
    d = 2048
    n = 256
    m = 50
    base_seed = 42
    num_samples = 5

    # k and b values to test
    k = 100  # Fixed value for k
    b = 30   # Fixed value for b

    # Paraphrasing intensities to evaluate
    paraphrasing_intensities = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Evaluate detection performance by paraphrasing intensity
    performance = evaluate_by_paraphrasing_intensity(
        model, tokenizer, d, k, b, n, m, base_seed, num_samples, paraphrasing_intensities, attack_type="substitution"
    )

    # Plot performance
    plot_paraphrasing_performance(paraphrasing_intensities, performance)

if __name__ == "__main__":
    run_experiment()


