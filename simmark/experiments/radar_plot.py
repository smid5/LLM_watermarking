# plots the distribution of the cost of each generation type and its translation-attacked text
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, test_distortion, detect, read_data

def get_robustness(method, num_tokens, filename):
    llm_config = load_llm_config('facebook/opt-125m')

    prompts = load_prompts(filename=filename)

    detection_name = method

    p_values_translated = test_watermark(
        prompts, num_tokens, llm_config, method, detection_name, "translate"
    )
    p_values_translated = np.array(p_values_translated)

    threshold = 1e-2
    true_positive_rate = np.mean(p_values_translated < threshold)

    print(f"Robustness for {method}: true positive rate = {true_positive_rate}")

    return true_positive_rate

def get_distortion_freeness(method, num_tokens, filename):
    llm_config = load_llm_config('facebook/opt-125m')

    prompts = load_prompts(filename=filename)

    detection_name = method

    perplexity_no_watermark = np.median(test_distortion(
        prompts, num_tokens, llm_config, "nomark", detection_name
    ))
    perplexity_watermarked = np.median(test_distortion(
        prompts, num_tokens, llm_config, method, detection_name
    ))

    # distortion_freeness = 1 / (1 + abs(perplexity_no_watermark - perplexity_watermarked) / perplexity_no_watermark)
    distortion_freeness = np.exp( -abs(perplexity_no_watermark - perplexity_watermarked) / perplexity_no_watermark)

    print(f"Distortion freeness for {method}: {distortion_freeness}")

    return distortion_freeness

def get_unforgeability(method, num_tokens, filename, num_inject=6, num_unwatermarked=6, folder="data/", seed=42):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)

    output_file = folder + f"{method}_forgeability.txt"
    detected_p_values = []

    p_values_watermarked = test_watermark(
        prompts, num_tokens, llm_config, generation_name=method, detection_name=method
    )
    # median_i = argmedian(p_values_watermarked)
    cache_file_watermarked = folder + f"{method}_{method}_.txt"
    cached_data_watermarked = read_data(cache_file_watermarked)

    p_values_unwatermarked = test_watermark(
    prompts, num_tokens, llm_config, generation_name="nomark", detection_name=method
    )
    cache_file_unwatermarked = folder + f"nomark_{method}_.txt"
    cached_data_unwatermarked = read_data(cache_file_unwatermarked)

    # median_text = extract_generated_text(prompts[median_i], cached_data_watermarked, num_tokens, seed)
    # median_text_list = median_text.split()
    # median_text_length = len(median_text_list)
    watermarked_text_lists = []
    for prompt in prompts:
        watermarked_text = extract_generated_text(prompt, cached_data_watermarked, num_tokens, seed)
        watermarked_text_list = watermarked_text.split()
        watermarked_text_lists.append(watermarked_text_list)

    for prompt in prompts:
        generated_text = extract_generated_text(prompt, cached_data_unwatermarked, num_tokens, seed)
        generated_text_list = generated_text.split()
        num_words = len(generated_text_list)
        
        for watermarked_text_list in watermarked_text_lists:
            watermarked_text_length = len(watermarked_text_list)
            w_idx = 0
            spoofed = generated_text_list[:num_unwatermarked]
            i = num_unwatermarked

            while i < num_words:
                if w_idx < watermarked_text_length:
                    spoofed.extend(watermarked_text_list[w_idx:w_idx+num_inject])
                    w_idx += num_inject
                
                spoofed.extend(generated_text_list[i:i+num_unwatermarked])
                i += num_unwatermarked

            spoofed_text = ' '.join(spoofed)
            detected_p_value = detect(spoofed_text, llm_config, method)
            detected_p_values.append(detected_p_value)
            output = {
                'prompt': prompt,
                'generated_text': spoofed_text,
                'p_value': detected_p_value,
                'seed' : seed,
                'num_tokens' : num_tokens,
            }
            with open(output_file, 'a') as f:
                f.write(str(output) + '\n')

    detected_p_values = np.array(detected_p_values)
    threshold = 1e-2
    true_negative_rate = np.mean(detected_p_values > threshold)

    print(f"Unforgeability for {method}: true_negative_rate={true_negative_rate}")
    log_file = "logs/unforgeability.txt"
    log_output = f"Unforgeability for {method}, {num_inject}, {num_unwatermarked}: true_negative_rate={true_negative_rate}"
    with open(log_file, 'a') as log_f:
        log_f.write(log_output + '\n')

    return true_negative_rate

def argmedian(arr):
    arr = np.array(arr)

    sorted_indices = np.argsort(arr)
    median_sorted_idx = len(arr) // 2
    median_index = sorted_indices[median_sorted_idx]

    print("Median index:", median_index)
    print("Median value:", arr[median_index])

    return median_index

def get_unforgeability_sadasivan(method, num_tokens, filename, folder="data/", seed=42):
    llm_config = load_llm_config('facebook/opt-125m')
    tokenizer = llm_config['tokenizer']
    prompts = load_prompts(filename=filename)
    cache_file = folder + f"nomark_{method}_.txt"
    cached_data = read_data(cache_file)

    output_file = folder + f"{method}_forgeability.txt"
    detected_p_values = []

    p_values = test_watermark(
        prompts, num_tokens, llm_config, generation_name="nomark", detection_name=method
    )
    min_i = np.argmin(p_values)
    min_p_value = np.min(p_values)
    worst_text = extract_generated_text(prompts[min_i], cached_data, num_tokens, seed)
    worst_ids = tokenizer.encode(worst_text, return_tensors="pt").squeeze()
    worst_ids_length = len(worst_ids)
    print(f"worst_ids_length={worst_ids_length}")

    for i, prompt in enumerate(prompts):
        if i != min_i:
            generated_text = extract_generated_text(prompt, cached_data, num_tokens, seed)
            ids = tokenizer.encode(generated_text, return_tensors="pt").squeeze()
            if worst_ids_length > len(ids):
                truncated_worst_text = tokenizer.decode(worst_ids[:len(ids)], skip_special_tokens=True)
            else:
                truncated_worst_text = worst_text

            spoofed_text = truncated_worst_text + " " + generated_text
            detected_p_value = detect(spoofed_text, llm_config, method)
            detected_p_values.append(detected_p_value)
            output = {
                'prompt': prompt,
                'generated_text': spoofed_text,
                'p_value': detected_p_value,
                'seed' : seed,
                'num_tokens' : num_tokens,
            }
            with open(output_file, 'a') as f:
                f.write(str(output) + '\n')

    detected_p_values = np.array(detected_p_values)
    threshold = 1e-2
    true_negative_rate = np.mean(detected_p_values > threshold)

    print(f"Unforgeability for {method}: min_p_value={min_p_value}, true_negative_rate={true_negative_rate}")

    return true_negative_rate

def extract_generated_text(prompt, cached_data, num_tokens, seed):
    matches = ['prompt', 'seed', 'num_tokens']

    # Find all indices where the prompt matches
    indices = [i for i, p in enumerate(cached_data['prompt']) if p == prompt]

    is_match = len(indices)>0
    # Check if any of those indices match seed and num_tokens
    for idx in indices:
        is_match = True
        for match in matches:
            if cached_data[match][idx] != locals()[match]:
                is_match = False
        if is_match:
            return cached_data['generated_text'][idx]
    return None

def normalize_scores(techniques, criteria, log=False, inverse=False):
    """
    log (Boolean) - whether to make it log scale or not
    inverse (Boolean) - True when lower score in techniques is better
    """
    if not log:
        scores = [technique[criteria] for technique in techniques.values()]
    else:
        scores = [np.log(technique[criteria]) for technique in techniques.values()]
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score
    if range_score == 0:
        normalized_scores = [0 for _ in scores]
    else:
        if not inverse:
            normalized_scores = [(score - min_score) / range_score for score in scores]
        else:
            normalized_scores = [(max_score - score) / range_score for score in scores]
    for technique, normalized_score in zip(techniques.values(), normalized_scores):
        technique[criteria] = normalized_score
    return techniques

def generate_radar_plot(k, b, num_tokens, filename, log=False):
    method_names = ["SimMark", "SoftRedList", "Unigram", "ExpMin", "HashlessExpMin", "SynthID"]
    methods = [f"simmark_{k}_{b}", "softred", "unigram", "expmin", "expminnohash", "synthid"]
    techniques = {}

    for method_name, method in zip(method_names, methods):
        robustness = get_robustness(method, num_tokens, filename)
        distortion_freeness = get_distortion_freeness(method, num_tokens, filename)
        unforgeability = get_unforgeability(method, num_tokens, filename)
        techniques[method_name] = {"robust": robustness, "distortion free": distortion_freeness, "unforgeability": unforgeability}
    
    if log:
        techniques = normalize_scores(techniques, "robust", log=True, inverse=False)
        techniques = normalize_scores(techniques, "distortion free", log=True, inverse=False)
        techniques = normalize_scores(techniques, "unforgeability", log=True, inverse=False)

    labels = list(techniques[method_names[0]].keys())
    num_vars = len(labels)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot each technique
    for name, scores in techniques.items():
        values = list(scores.values())
        values += values[:1]  # repeat first value to close the plot
        ax.plot(angles, values, label=name)
        ax.fill(angles, values, alpha=0.1)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set y-label range
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])

    # Add legend and title
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Watermarking Technique Comparison", size=15)
    plt.tight_layout()

    if not log:
        plt.savefig(f"Figures/radar_{k}_{b}.pdf")
    else:
        plt.savefig(f"Figures/radar_log_{k}_{b}.pdf")
    plt.close()

def generate_radar_comparison(kb_list, num_tokens, filename):
    """
    kb_list should be a list of tuples (k,b)
    """
    method_names = []
    methods = []
    for k, b in kb_list:
        method_names.append(f"SimMark{k}-{b}")
        methods.append(f"simmark_{k}_{b}")
    method_names += ["SoftRedList", "Unigram", "ExpMin", "HashlessExpMin", "SynthID"]
    methods += ["softred", "unigram", "expmin", "expminnohash", "synthid"]
    techniques = {}

    for method_name, method in zip(method_names, methods):
        robustness = get_robustness(method, num_tokens, filename)
        distortion_freeness = get_distortion_freeness(method, num_tokens, filename)
        unforgeability = get_unforgeability(method, num_tokens, filename)
        techniques[method_name] = {"robust": robustness, "distortion free": distortion_freeness, "unforgeability": unforgeability}

    labels = list(techniques[method_names[0]].keys())
    num_vars = len(labels)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot each technique
    for name, scores in techniques.items():
        values = list(scores.values())
        values += values[:1]  # repeat first value to close the plot
        ax.plot(angles, values, label=name)
        ax.fill(angles, values, alpha=0.1)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set y-label range
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])

    # Add legend and title
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Watermarking Technique Comparison", size=15)
    plt.tight_layout()

    plt.savefig("Figures/radar_comparison.pdf")
    plt.close()