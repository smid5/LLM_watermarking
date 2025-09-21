# plots the distribution of the cost of each generation type
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

from .utils import load_llm_config, test_distortion, load_prompts, METHODS, COLORS, KEYS, LINESTYLES
from collections import defaultdict
import numpy as np

def plot_sentence_length_median_distortion(sentence_lengths, distortions, filename):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(10, 4.5))  # wider plot

    for (label, values) in distortions.items():
        if isinstance(label, tuple):
            method_name, key_name = label
            color = COLORS[method_name]
            linestyle = LINESTYLES[key_name]
            legend_label = f"{method_name} ({key_name})"
        else:
            method_name = label
            color = COLORS[method_name]
            linestyle = "-"
            legend_label = method_name

        plt.plot(
            sentence_lengths,
            values,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1,
            linestyle=linestyle,
            color=color,
            linewidth=2,
            label=legend_label
        )

    # Labels and ticks
    plt.xlabel("Sentence Length", fontsize=14)
    plt.ylabel("Median Distortion", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Grid
    plt.grid(True, linestyle="--", alpha=0.6)

    # Legend outside
    plt.legend(fontsize=11, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def sentence_length_median_distortion(length_variations, filename, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["Standard Hashing", "SimHash"], k=4, b=4, seeds=[42], model_name='meta-llama/Meta-Llama-3-8B'):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)

    distortions = defaultdict(dict)
    if "No Watermark" not in method_names:
        method_names.append("No Watermark")

    for length in length_variations:
        applicable_prompts = [p for p in prompts if len(p.split()) < length]
        if not applicable_prompts:
            continue

        for method_name in method_names:
            if method_name == "No Watermark":
                method = "nomark"
                detection_name = f"expmin_simhash"
                distortion_vals = [test_distortion(
                    applicable_prompts, length, llm_config, method, detection_name, seed=seed
                ) for seed in seeds]
                median_distortion = np.median(distortion_vals)
                distortions[method_name][length] = median_distortion

            else:
                for key_name in key_names:
                    method = f"{METHODS[method_name]}_{KEYS[key_name]}"

                    distortion_vals = [test_distortion(
                        applicable_prompts, length, llm_config, method, method, seed=seed
                    ) for seed in seeds]
                    median_distortion = np.median(distortion_vals)
                    distortions[(method_name, key_name)][length] = median_distortion
        
    sorted_lengths = sorted(distortions["No Watermark"].keys())
    for key in distortions:
        distortions[key] = [distortions[key][l] for l in sorted_lengths]
    plot_sentence_length_median_distortion(sorted_lengths, distortions, f"Figures/sentence_length_vs_distortion.pdf")

def plot_distortion_dist(num_tokens, filename, k=4, b=4):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    method_names = {"SimMark", "ExpMin", "SoftRedList", "Unigram", "SynthID", "No Watermark"}

    perplexity = {}

    for method_name in method_names:
        method = f"simmark_{k}_{b}" if method_name == "SimMark" else METHODS[method_name]
        detection_name = f"simmark_{k}_{b}" if method == "nomark" else method


        perplexity[method_name]= test_distortion(
            prompts, num_tokens, llm_config, method, detection_name
        )

    plt.style.use(['science'])
    plt.figure(figsize=(4, 3))

    # Labels and legend
    plt.xscale("linear")
    for idx, key in enumerate(perplexity):
        sns.kdeplot(perplexity[key], label=key, log_scale=False, linewidth=2, color=COLORS[key], cut=0)
    
    plt.xlabel("Perplexity")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig(f"figures/perplexity_dist_{num_tokens}.pdf")

    # Show the plot