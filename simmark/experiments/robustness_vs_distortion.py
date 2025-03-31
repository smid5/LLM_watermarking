import numpy as np
import matplotlib.pyplot as plt

from .utils import load_llm_config, load_prompts, test_watermark, test_distortion

def plot_robustness_vs_distortion(robustness, distortion, filename, k, b):
    methods = list(robustness.keys())
    robustness_values = list(robustness.values())
    distortion_values = list(distortion.values())

    plt.figure(figsize=(8, 6))
    for i, key in enumerate(methods):
        plt.scatter(robustness_values[i], distortion_values[i], label=key)

    # Labels and legend
    plt.xlabel('Robustness (average p-value on translated text)')
    plt.ylabel('Distortion (perplexity)')
    plt.title(f'Robustness vs. Distortion for k={k}, b={b}')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def generate_robustness_vs_distortion(filename, num_tokens, k=5, b=8):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)

    robustness = {}
    distortion = {}
    generation_methods = [f"simmark_{k}_{b}", "unigram", "softred", "expmin", "synthid", "expminnohash", "nomark"]
    detection_methods = [f"simmark_{k}_{b}", "unigram", "softred", "expmin", "synthid", "expminnohash", f"simmark_{k}_{b}"]

    for generation_method, detection_method in zip(generation_methods, detection_methods):
        robustness[generation_method] = np.mean(test_watermark(prompts, num_tokens, llm_config, generation_method, detection_method, "translate"))
        distortion[generation_method] = np.mean(test_distortion(prompts, num_tokens, llm_config, generation_method))

    plot_robustness_vs_distortion(robustness, distortion, f"Figures/robustness_distortion_k{k}_b{b}.pdf", k, b)
