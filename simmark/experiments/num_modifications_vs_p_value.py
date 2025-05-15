import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import test_watermark, load_llm_config, load_prompts, modify_text, cbcolors, COLORS

def plot_p_value_modifications(modifications, p_values_dict, filename, xlabel):
    plt.style.use(['science'])
    plt.figure(figsize=(6, 4))
    
    # Iterate through each method and plot its data
    for idx, (label, values) in enumerate(p_values_dict.items()):
        plt.plot(modifications, values, marker='o', linestyle='-', color=COLORS[label], linewidth=2, label=label)

    plt.yscale("linear")  
    plt.xlabel(xlabel)
    plt.ylabel("Percent with p-value below .01")
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
        threshold = 1e-2
        p_values["SimMark"][i] = np.mean(np.array(test_watermark(
            prompts, num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}", f"{attack}_{num_modify}"
        ))<threshold)*100

        p_values["Unigram"][i] = np.mean(np.array(test_watermark(
            prompts, num_tokens, llm_config, "unigram", "unigram", f"{attack}_{num_modify}"
        ))<threshold)*100

        p_values["SoftRedList"][i] = np.mean(np.array(test_watermark(
            prompts, num_tokens, llm_config, "softred", "softred", f"{attack}_{num_modify}"
        ))<threshold)*100

        p_values["ExpMin"][i] = np.mean(np.array(test_watermark(
            prompts, num_tokens, llm_config, "expmin", "expmin", f"{attack}_{num_modify}"
        ))<threshold)*100

        p_values["SynthID"][i] = np.mean(np.array(test_watermark(
            prompts, num_tokens, llm_config, "synthid", "synthid", f"{attack}_{num_modify}"
        ))<threshold)*100

    save_filename = f"figures/p_value_vs_{attack}_attack_k{k}_b{b}.pdf"
    if attack=="duplicate":
        xlabel="Number of Duplicate Word Insertions"
    elif attack=="modify":
        xlabel="Number of Unrelated Word Replacements"
    elif attack=="translate":
        xlabel="Number of Translated Word Replacements"
    else:
        xlabel="Number of Word Modifications"
    # Generate plot
    plot_p_value_modifications(modifications, p_values, save_filename, xlabel)