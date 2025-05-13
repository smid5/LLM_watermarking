import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from .utils import load_llm_config, load_prompts, test_watermark, test_distortion, COLORS

def plot_robustness_vs_distortion(robustness, distortion, filename, num_tokens, attack_name):
    methods = list(robustness.keys())
    robustness_values = list(robustness.values())
    distortion_values = list(distortion.values())

    plt.figure(figsize=(8, 6))
    plt.style.use(['science'])

    # Scatter plot with correct labeling
    for i, key in enumerate(methods):
        xs = robustness_values[i]
        if xs == 0:
            xs = 1e-20
        ys = distortion_values[i]

        plt.scatter(xs, ys, label=key, color=COLORS[key])

    plt.xscale("log")

    # Labels and legends
    plt.xlabel('Robustness (median p-value)')
    plt.ylabel('Distortion (median perplexity)')
    if not attack_name:
        plt.title(f'Robustness vs. Distortion for Watermarked text, n={num_tokens}')
    elif attack_name=="translate": 
        plt.title(f'Robustness vs. Distortion for Translated text, n={num_tokens}')
    
    plt.legend()

    # Show and save
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def generate_robustness_vs_distortion(filename, num_tokens, k=4, b=4, attack_name=""):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)

    robustness = {}
    distortion = {}
    method_names = ["SimMark", "Unigram", "SoftRedList", "SynthID", "ExpMin", "No Watermark"]
    generation_methods = [f"simmark_{k}_{b}", "unigram", "softred", "synthid", "expmin", "nomark"]
    detection_methods = [f"simmark_{k}_{b}", "unigram", "softred", "synthid", "expmin", f"simmark_{k}_{b}"]

    for method_name, generation_method, detection_method in zip(method_names, generation_methods, detection_methods):
        robustness[method_name] = np.median(test_watermark(prompts, num_tokens, llm_config, generation_method, detection_method, attack_name))
        distortion[method_name] = np.median(test_distortion(prompts, num_tokens, llm_config, generation_method, detection_method, attack_name))

    plot_robustness_vs_distortion(robustness, distortion, f"Figures/robustness_distortion_k{k}_b{b}_{num_tokens}_{attack_name}.pdf", num_tokens, attack_name)