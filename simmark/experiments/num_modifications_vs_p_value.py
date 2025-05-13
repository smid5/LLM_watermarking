import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import test_watermark, load_llm_config, load_prompts, modify_text, cbcolors, COLORS

def plot_p_value_modifications(modifications, p_values_dict, filename, title):
    plt.style.use(['science'])
    plt.figure(figsize=(6, 4))
    
    # Iterate through each method and plot its data
    for idx, (label, values) in enumerate(p_values_dict.items()):
        plt.plot(modifications, values, marker='o', linestyle='-', color=COLORS[label], linewidth=2, label=label)

    plt.yscale("log")  # Set y-axis to log scale for better visualization
    plt.xlabel("Number of Modifications")
    plt.ylabel("Median p-Value")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(modifications)
    plt.savefig(filename)
    plt.close()

def generate_p_value_modification_experiment(filename, attack, k=4, b=4, modification_values = list(range(0, 31, 3)), num_tokens=100):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    modifications = np.array(modification_values)
    num_modifications = len(modification_values)

    # Dictionary to store p-values for each method
    p_values = {
        "SimMark": np.zeros(num_modifications),
        "Unigram": np.zeros(num_modifications),
        "SoftRedList": np.zeros(num_modifications),
        "ExpMin": np.zeros(num_modifications),
        "SynthID": np.zeros(num_modifications)
    }

    for i, num_modify in enumerate(modification_values):

        # Compute p-values for each method
        p_values["SimMark"][i] = np.median(test_watermark(
            prompts, num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}", f"{attack}_{num_modify}"
        ))

        p_values["Unigram"][i] = np.median(test_watermark(
            prompts, num_tokens, llm_config, "unigram", "unigram", f"{attack}_{num_modify}"
        ))

        p_values["SoftRedList"][i] = np.median(test_watermark(
            prompts, num_tokens, llm_config, "softred", "softred", f"{attack}_{num_modify}"
        ))

        p_values["ExpMin"][i] = np.median(test_watermark(
            prompts, num_tokens, llm_config, "expmin", "expmin", f"{attack}_{num_modify}"
        ))

        p_values["SynthID"][i] = np.median(test_watermark(
            prompts, num_tokens, llm_config, "synthid", "synthid", f"{attack}_{num_modify}"
        ))

    if attack=="modify":
        title = "Effect of Modifications on p-Values"
        save_filename = f"figures/p_value_vs_modifications_k{k}_b{b}.pdf"
    elif attack=="translate":
        title = "Effect of Translation Modifications on p-Values"
        save_filename = f"figures/p_value_vs_translation_modifications_k{k}_b{b}.pdf"
    elif attack=="mask":
        title = "Effect of Mask Modifications on p-Values"
        save_filename = f"figures/p_value_vs_mask_modifications_k{k}_b{b}.pdf"
    elif attack=="delete":
        title = "Effect of Deletions on p-Values"
        save_filename = f"figures/p_value_vs_deletions_k{k}_b{b}.pdf"
    elif attack=="insert":
        title = "Effect of Insertions on p-Values"
        save_filename = f"figures/p_value_vs_insertions_k{k}_b{b}.pdf"
    elif attack=="duplicate":
        title = "Effect of Duplicate Insertions on p-Values"
        save_filename = f"figures/p_value_vs_duplications_k{k}_b{b}.pdf"
    # Generate plot
    plot_p_value_modifications(modifications, p_values, save_filename, title)