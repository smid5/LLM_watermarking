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

def generate_p_value_modification_experiment(filename, attack, k=4, b=4, modification_values = list(range(0, 31, 3)), num_tokens=100, seeds=[42]):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    num_prompts = len(prompts)
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
    method_names = ["SimMark", "Unigram", "SoftRedList", "ExpMin", "SynthID"]
    methods = [f"simmark_{k}_{b}", "unigram", "softred", "expmin", "synthid"]

    for i, num_modify in enumerate(modification_values):

        # Compute p-values for each method
        threshold = 1e-2

        for method_name, method in zip(method_names, methods):
            if method_name=="SimMark" or method_name=="ExpMin":
                results = np.empty((0, num_prompts))
                for seed in seeds:
                    new_data = np.array(test_watermark(
                        prompts, num_tokens, llm_config, method, method, f"{attack}_{num_modify}", seed=seed
                    ))
                    results = np.vstack([results, new_data])
            else:
                results = np.array(test_watermark(
                        prompts, num_tokens, llm_config, method, method, f"{attack}_{num_modify}", seed=42
                    ))
            p_values[method_name][i] = np.mean(results<threshold)*100

    save_filename = f"figures/p_value_vs_{attack}_attack_k{k}_b{b}.pdf"
    if attack=="duplicate":
        xlabel="Number of Related Word Insertions"
    elif attack=="modify":
        xlabel="Number of Unrelated Word Replacements"
    elif attack=="translate":
        xlabel="Number of Translated Word Replacements"
    else:
        xlabel="Number of Word Modifications"
    # Generate plot
    plot_p_value_modifications(modifications, p_values, save_filename, xlabel)

def subplot_p_value_modifications(modifications, p_values_dict, ax, xlabel):
    for idx, (label, values) in enumerate(p_values_dict.items()):
        ax.plot(modifications, values, marker='o', linestyle='-', color=COLORS[label], linewidth=2, label=label)

    ax.set_yscale("linear")
    ax.set_xlabel(xlabel, fontsize=24)
    ax.legend()
    ax.grid()
    ax.set_xticks(modifications)
    ax.tick_params(axis='x', labelsize=20) 
    ax.tick_params(axis='y', labelsize=20) 
    ax.set_ylim(15, 105)  
    ax.set_yticks(np.arange(20, 101, 20)) 

def generate_p_value_modification_subplot(filename, attack, ax, k=4, b=4, modification_values = list(range(0, 31, 3)), num_tokens=100, seeds=[42]):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    num_prompts = len(prompts)
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
    method_names = ["SimMark", "Unigram", "SoftRedList", "ExpMin", "SynthID"]
    methods = [f"simmark_{k}_{b}", "unigram", "softred", "expmin", "synthid"]

    for i, num_modify in enumerate(modification_values):

        # Compute p-values for each method
        threshold = 1e-2

        for method_name, method in zip(method_names, methods):
            if method_name=="SimMark" or method_name=="ExpMin":
                results = np.empty((0, num_prompts))
                for seed in seeds:
                    new_data = np.array(test_watermark(
                        prompts, num_tokens, llm_config, method, method, f"{attack}_{num_modify}", seed=seed
                    ))
                    results = np.vstack([results, new_data])
            else:
                results = np.array(test_watermark(
                        prompts, num_tokens, llm_config, method, method, f"{attack}_{num_modify}", seed=42
                    ))
            p_values[method_name][i] = np.mean(results<threshold)*100

    save_filename = f"figures/p_value_vs_{attack}_attack_k{k}_b{b}.pdf"
    if attack=="duplicate":
        xlabel="Number of Related Word Insertions"
    elif attack=="modify":
        xlabel="Number of Unrelated Token Substitutions"
    elif attack=="translate":
        xlabel="Number of Related Token Substitutions"
    else:
        xlabel="Number of Word Modifications"
    # Generate plot
    subplot_p_value_modifications(modifications, p_values, ax, xlabel)

def plot_modification_comparison(filename, attacks, k=4, b=4, modification_values = list(range(0, 31, 3)), num_tokens=100, seeds=[42]):
    plt.style.use(['science'])

    # Create a figure with 3 subplots horizontally
    fig, axs = plt.subplots(1, 3, figsize=(18, 5)) 

    for i, attack in enumerate(attacks):
        generate_p_value_modification_subplot(filename, attack, axs[i], k=k, b=b, modification_values=modification_values, num_tokens=num_tokens, seeds=seeds)

    handles, labels = axs[0].get_legend_handles_labels()

    # Remove legends from individual subplots
    for ax in axs:
        ax.legend_.remove()

    # Add a single legend below the plots
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=24)

    # Add y-axis label to the leftmost subplot
    axs[0].set_ylabel("Percent with p-value below .01", fontsize=22)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"figures/p_value_vs_attack_combined_k{k}_b{b}.pdf")
    plt.close()