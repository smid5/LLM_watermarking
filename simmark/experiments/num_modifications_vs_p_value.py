import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import test_watermark, load_llm_config, load_prompts, modify_text, cbcolors, linestyles

def plot_p_value_modifications(modifications, p_values_dict, filename):
    plt.style.use(['science'])
    plt.figure(figsize=(6, 4))
    
    # Iterate through each method and plot its data
    for idx, (label, values) in enumerate(p_values_dict.items()):
        plt.plot(modifications, values, marker='o', linestyle='-', color=cbcolors[idx], linewidth=2, label=label)

    plt.yscale("log")  # Set y-axis to log scale for better visualization
    plt.xlabel("Number of Modifications")
    plt.ylabel("Median p-Value")
    plt.title("Effect of Modifications on p-Values")
    plt.legend()
    plt.grid()
    plt.xticks(modifications)
    plt.savefig(filename)
    plt.close()

def generate_p_value_modification_experiment(filename, k=2, b=64, num_modifications=21, num_tokens=100):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    modifications = np.arange(num_modifications)

    # Dictionary to store p-values for each method
    p_values = {
        "SimMark": np.zeros(num_modifications),
        "Unigram": np.zeros(num_modifications),
        "SoftRedList": np.zeros(num_modifications),
        "ExpMin": np.zeros(num_modifications),
        "ExpMinNoHash": np.zeros(num_modifications),
        "SynthID": np.zeros(num_modifications)
    }

    for i in range(num_modifications):

        # Compute p-values for each method
        p_values["SimMark"][i] = np.mean(test_watermark(
            prompts, num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}", f"modify_{i}"
        ))

        p_values["Unigram"][i] = np.mean(test_watermark(
            prompts, num_tokens, llm_config, "unigram", "unigram", f"modify_{i}"
        ))

        p_values["SoftRedList"][i] = np.mean(test_watermark(
            prompts, num_tokens, llm_config, "softred", "softred", f"modify_{i}"
        ))

        p_values["ExpMin"][i] = np.mean(test_watermark(
            prompts, num_tokens, llm_config, "expmin", "expmin", f"modify_{i}"
        ))

        p_values["ExpMinNoHash"][i] = np.mean(test_watermark(
            prompts, num_tokens, llm_config, "expminnohash", "expminnohash", f"modify_{i}"
        ))

        p_values["SynthID"][i] = np.mean(test_watermark(
            prompts, num_tokens, llm_config, "synthid", "synthid", f"modify_{i}"
        ))

    # Generate plot
    plot_p_value_modifications(modifications, p_values, f"figures/p_value_vs_modifications_k{k}_b{b}.pdf")

# if __name__ == '__main__':
#     generate_p_value_modification_experiment("sentence_starters.txt")