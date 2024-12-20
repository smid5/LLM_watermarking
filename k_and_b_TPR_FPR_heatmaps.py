import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash
from detection_simhash import simhash_detect_with_permutation


# Function to generate paraphrased text
def paraphrase_text(text):
    return text.replace("a", "e")  # Simple paraphrasing logic


# Function to generate unrelated text
def generate_unrelated_text(model, tokenizer, seed):
    torch.manual_seed(seed)
    random_topics = [
        "astronomy", "cooking", "robotics", "music theory",
        "ancient history", "quantum mechanics", "gardening", "sports analysis"
    ]
    random_topic = random_topics[seed % len(random_topics)]
    random_prompt = f"Write a brief introduction to {random_topic}."
    prompts = tokenizer(random_prompt, return_tensors="pt").input_ids
    outputs = model.generate(prompts, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Function to calculate TPR and FPR
def evaluate_detection(model, tokenizer, context, prompts, d, k_values, b_values, n, m, seed, threshold=0.5, num_samples=5):
    tpr_results = np.zeros((len(k_values), len(b_values)))
    fpr_results = np.zeros((len(k_values), len(b_values)))

    for i, k in enumerate(k_values):
        for j, b in enumerate(b_values):
            print(k)
            print(b)
            print("")
            tp, fp = 0, 0  # True positives, false positives
            total_watermarked, total_paraphrased, total_unrelated = num_samples, num_samples, num_samples

            for sample in range(num_samples):
                sample_seed = seed + sample

                # Generate watermarked text
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
                wm_p_value, wm_result, _ = simhash_detect_with_permutation(
                    context=context,
                    observed_token=watermarked_tokens[-1],
                    d=d,
                    k=k,
                    b=b,
                    seed=sample_seed,
                    model=model,
                    tokenizer=tokenizer,
                    n_runs=100,
                    threshold=threshold,
                )
                if wm_result == "Watermark detected":
                    tp += 1

                # Generate paraphrased text
                watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
                paraphrased_text = paraphrase_text(watermarked_text)
                paraphrased_tokens = tokenizer(paraphrased_text, return_tensors="pt").input_ids[0]
                para_p_value, para_result, _ = simhash_detect_with_permutation(
                    context=context,
                    observed_token=int(paraphrased_tokens[-1]),
                    d=d,
                    k=k,
                    b=b,
                    seed=sample_seed,
                    model=model,
                    tokenizer=tokenizer,
                    n_runs=100,
                    threshold=threshold,
                )
                if para_result == "Watermark detected":
                    tp += 1

                # Generate unrelated text
                unrelated_text = generate_unrelated_text(model, tokenizer, sample_seed)
                unrelated_tokens = tokenizer(unrelated_text, return_tensors="pt").input_ids[0]
                unrelated_p_value, unrelated_result, _ = simhash_detect_with_permutation(
                    context=context,
                    observed_token=int(unrelated_tokens[-1]),
                    d=d,
                    k=k,
                    b=b,
                    seed=sample_seed,
                    model=model,
                    tokenizer=tokenizer,
                    n_runs=100,
                    threshold=threshold,
                )
                if unrelated_result == "Watermark detected":
                    fp += 1

            # Calculate TPR and FPR
            tpr_results[i, j] = tp / (total_watermarked + total_paraphrased)
            fpr_results[i, j] = fp / total_unrelated

    return tpr_results, fpr_results


# Plot heatmaps
def plot_heatmap(data, k_values, b_values, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap="viridis", aspect="auto")
    plt.colorbar(label="Rate")
    plt.xticks(ticks=np.arange(len(b_values)), labels=b_values)
    plt.yticks(ticks=np.arange(len(k_values)), labels=k_values)
    plt.xlabel("b (Bits per Hash)")
    plt.ylabel("k (Number of Hash Functions)")
    plt.title(title)
    plt.show()


# Main function to run experiments
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
    threshold = 0.5
    num_samples = 5

    # k and b values to test
    k_values = [100, 150]
    b_values = [30, 40]

    # Input context
    context = "Once upon a time, in a faraway land, a boy's heart's aching."
    prompts = tokenizer(context, return_tensors="pt").input_ids

    # Evaluate detection
    tpr_results, fpr_results = evaluate_detection(
        model, tokenizer, context, prompts, d, k_values, b_values, n, m, base_seed, threshold, num_samples
    )

    # Plot heatmaps
    plot_heatmap(tpr_results, k_values, b_values, "TPR Heatmap for Paraphrased and Watermarked Text")
    plot_heatmap(fpr_results, k_values, b_values, "FPR Heatmap for Unrelated Text")


if __name__ == "__main__":
    run_experiment()

