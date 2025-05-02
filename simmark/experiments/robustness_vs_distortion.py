import numpy as np
import matplotlib.pyplot as plt

from .utils import load_llm_config, load_prompts, test_watermark, test_distortion

def plot_robustness_vs_distortion(robustness, distortion, filename, k, b, num_tokens, attack_name):
    methods = list(robustness.keys())
    robustness_values = list(robustness.values())
    distortion_values = list(distortion.values())

    plt.figure(figsize=(8, 6))

    # Scatter plot with correct labeling
    for i, key in enumerate(methods):
        xs = robustness_values[i]
        if xs == 0:
            xs = 1e-20
        ys = distortion_values[i]

        plt.scatter(xs, ys, label=key)

    plt.xscale("log")

    # Labels and legends
    plt.xlabel('Robustness (median p-value)')
    plt.ylabel('Distortion (median perplexity)')
    if not attack_name:
        plt.title(f'Robustness vs. Distortion for Watermarked text, k={k}, b={b}, n={num_tokens}')
    elif attack_name=="translate": 
        plt.title(f'Robustness vs. Distortion for Translated text, k={k}, b={b}, n={num_tokens}')
    
    plt.legend()

    # Show and save
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def generate_robustness_vs_distortion(filename, num_tokens, k=5, b=8, attack_name=""):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)

    robustness = {}
    distortion = {}
    generation_methods = [f"simmark_{k}_{b}", "unigram", "softred", "synthid", "expmin", "expminnohash", "nomark"]
    detection_methods = [f"simmark_{k}_{b}", "unigram", "softred", "synthid", "expmin", "expminnohash", f"simmark_{k}_{b}"]

    for generation_method, detection_method in zip(generation_methods, detection_methods):
        robustness[generation_method] = np.median(test_watermark(prompts, num_tokens, llm_config, generation_method, detection_method, attack_name))
        distortion[generation_method] = np.median(test_distortion(prompts, num_tokens, llm_config, generation_method, detection_method, attack_name))

    plot_robustness_vs_distortion(robustness, distortion, f"Figures/robustness_distortion_k{k}_b{b}_{num_tokens}_{attack_name}.pdf", k, b, num_tokens, attack_name)