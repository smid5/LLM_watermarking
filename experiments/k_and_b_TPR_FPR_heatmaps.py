import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_simhash import generate_with_simhash
from detection_simhash import simhash_detect_with_permutation
from transformers import MarianMTModel, MarianTokenizer

# Function to generate paraphrased text
def paraphrase_text(text):
    return text.replace("a", "e")  # Simple paraphrasing logic

def back_translate(text, source_lang="en", pivot_lang="fr"):
    """
    Performs back-translation: text → pivot_lang → text.

    Args:
        text (str): Input text to paraphrase.
        source_lang (str): Source language code (e.g., "en").
        pivot_lang (str): Pivot language code (e.g., "fr").
    
    Returns:
        str: Back-translated text.
    """
    # Load source → pivot translation model and tokenizer
    source_to_pivot_model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{pivot_lang}"
    pivot_to_source_model_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-{source_lang}"

    # Initialize models and tokenizers
    source_to_pivot_tokenizer = MarianTokenizer.from_pretrained(source_to_pivot_model_name)
    source_to_pivot_model = MarianMTModel.from_pretrained(source_to_pivot_model_name)
    pivot_to_source_tokenizer = MarianTokenizer.from_pretrained(pivot_to_source_model_name)
    pivot_to_source_model = MarianMTModel.from_pretrained(pivot_to_source_model_name)

    # Step 1: Translate source → pivot
    inputs = source_to_pivot_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    pivot_translation = source_to_pivot_model.generate(**inputs)
    pivot_text = source_to_pivot_tokenizer.decode(pivot_translation[0], skip_special_tokens=True)

    # Step 2: Translate pivot → source
    inputs = pivot_to_source_tokenizer(pivot_text, return_tensors="pt", max_length=512, truncation=True)
    back_translation = pivot_to_source_model.generate(**inputs)
    paraphrased_text = pivot_to_source_tokenizer.decode(back_translation[0], skip_special_tokens=True)

    return paraphrased_text

def paraphrase_text_backtranslation(tokens, tokenizer):
    """
    Applies back-translation to paraphrase text.

    Args:
        tokens (torch.Tensor): Input tokens.
        tokenizer: Tokenizer for decoding and encoding text.
    
    Returns:
        torch.Tensor: Paraphrased tokens.
    """
    # Decode tokens to text
    original_text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Apply back-translation
    paraphrased_text = back_translate(original_text)

    # Re-encode paraphrased text to tokens
    paraphrased_tokens = tokenizer(paraphrased_text, return_tensors="pt").input_ids[0]
    return paraphrased_tokens

# Function to generate unrelated and unwatermarked text dynamically
def generate_unrelated_text(model, tokenizer, m, seed):
    torch.manual_seed(seed)
    prompts = tokenizer("", return_tensors="pt").input_ids  # Start with an empty prompt
    outputs = model.generate(prompts, max_length=m, num_return_sequences=1, do_sample=True, top_k=50, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to evaluate detection performance
def evaluate_detection(model, tokenizer, k_values, b_values, n, m, seed, threshold=0.5, num_samples=5):
    tpr_results = np.zeros((len(k_values), len(b_values)))
    fpr_results = np.zeros((len(k_values), len(b_values)))

    for i, k in enumerate(k_values):
        for j, b in enumerate(b_values):
            print(f"Testing with k={k}, b={b}\n")
            tp, fp = 0, 0  # True positives, false positives
            total_watermarked, total_paraphrased, total_unrelated = num_samples, num_samples, num_samples

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

                wm_p_value, wm_result, _ = simhash_detect_with_permutation(
                    context=tokenizer.decode(watermarked_tokens[:-1], skip_special_tokens=True),
                    observed_token=watermarked_tokens[-1],
                    vocab_size=tokenizer.vocab_size,  # Add vocab_size
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

                print("Watermarked p-value: ")
                print(wm_p_value)
                
                # Paraphrase text using back-translation
                paraphrased_tokens = paraphrase_text_backtranslation(watermarked_tokens, tokenizer)

                # Detect watermark in paraphrased text
                paraphrased_text = tokenizer.decode(paraphrased_tokens, skip_special_tokens=True)
                para_p_value, para_result, _ = simhash_detect_with_permutation(
                    context=tokenizer.decode(watermarked_tokens[:-1], skip_special_tokens=True),
                    observed_token=int(paraphrased_tokens[-1]),
                    vocab_size=tokenizer.vocab_size,
                    k=k,
                    b=b,
                    seed=sample_seed,
                    model=model,
                    tokenizer=tokenizer,
                    n_runs=100,
                    threshold=threshold,
                )
                
                # Update detection counts
                if para_result == "Watermark detected":
                    tp += 1

                print("Paraphrase p value: ")
                print(para_p_value)

                # Generate paraphrased text
                # watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
                # paraphrased_text = paraphrase_text(watermarked_text)
                # paraphrased_tokens = tokenizer(paraphrased_text, return_tensors="pt").input_ids[0]
                # para_p_value, para_result, _ = simhash_detect_with_permutation(
                #     context=tokenizer.decode(watermarked_tokens[:-1], skip_special_tokens=True),
                #     observed_token=int(paraphrased_tokens[-1]),
                #     vocab_size=tokenizer.vocab_size,  # Add vocab_size
                #     k=k,
                #     b=b,
                #     seed=sample_seed,
                #     model=model,
                #     tokenizer=tokenizer,
                #     n_runs=100,
                #     threshold=threshold,
                # )
                # if para_result == "Watermark detected":
                #     tp += 1

                # print("Paraphrase p-value: ")
                # print(para_p_value)

                # Generate unrelated text
                unrelated_text = generate_unrelated_text(model, tokenizer, m, sample_seed)
                unrelated_tokens = tokenizer(unrelated_text, return_tensors="pt").input_ids[0]
                unrelated_p_value, unrelated_result, _ = simhash_detect_with_permutation(
                    context=tokenizer.decode(watermarked_tokens[:-1], skip_special_tokens=True),
                    observed_token=int(unrelated_tokens[-1]),
                    vocab_size=tokenizer.vocab_size,  # Add vocab_size
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

                print("Unrelated p value: ")
                print(unrelated_p_value)

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

    plt.savefig(str(title) + ".png") 

    plt.show()

# Main function to run experiments
def run_experiment():
    # Model and tokenizer setup
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Parameters
    n = 256
    m = 50
    base_seed = 42
    threshold = 0.1
    num_samples = 5

    # k and b values to test
    k_values = [100, 150]
    b_values = [30, 40]

    # Evaluate detection
    tpr_results, fpr_results = evaluate_detection(
        model, tokenizer, k_values, b_values, n, m, base_seed, threshold, num_samples
    )

    # Plot heatmaps
    plot_heatmap(tpr_results, k_values, b_values, "TPR Heatmap for Paraphrased and Watermarked Text")
    plot_heatmap(fpr_results, k_values, b_values, "FPR Heatmap for Unrelated Text")

if __name__ == "__main__":
    run_experiment()
