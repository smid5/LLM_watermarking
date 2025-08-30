# plots the distribution of the cost of each generation type
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

from .utils import load_llm_config, test_distortion, load_prompts, METHODS, COLORS

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