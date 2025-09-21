import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, METHODS, COLORS, KEYS, LINESTYLES
from collections import defaultdict
from .tpr import compute_tpr


def plot_tpr_modifications(modifications, tprs, filename, xlabel, fpr):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(10, 4.5))

    for (label, values) in tprs.items():
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
            modifications,
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
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(f"TPR under fixed FPR of {fpr}", fontsize=14)
    plt.xticks(modifications, fontsize=12)
    plt.yticks(fontsize=12)

    # Grid
    plt.grid(True, linestyle="--", alpha=0.6)

    # Legend outside
    plt.legend(fontsize=11, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def generate_tpr_modification_experiment(modification_values, num_tokens, filename, attack_name, method_names=["ExpMin", "SynthID", "WaterMax"], key_names=["Standard Hashing", "SimHash"], fpr= 1e-3, k=4, b=4, wm_seeds=[42], unwm_seeds=[42], model_name='meta-llama/Meta-Llama-3-8B'):
    llm_config = load_llm_config(model_name)
    prompts = load_prompts(filename=filename)
    modifications = np.array(modification_values)

    tprs = defaultdict(list)
    pvals_unwm = defaultdict(list)

    for method_name in method_names:
        for key_name in key_names:
            if method_name == "WaterMax":
                method = f"{METHODS[method_name]}_{KEYS[key_name]}_{4}_{8}"
            else:
                method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
            pvals_unwm[(method_name, key_name)] = [test_watermark(
                prompts, num_tokens, llm_config, "nomark", method, seed=seed
            ) for seed in unwm_seeds]
            
    for num_modify in modification_values:
        for method_name in method_names:
            for key_name in key_names:
                if method_name == "WaterMax":
                    method = f"{METHODS[method_name]}_{KEYS[key_name]}_{4}_{8}"
                else:
                    method = f"{METHODS[method_name]}_{KEYS[key_name]}_{k}_{b}"
                print(f"Evaluating {method} with {attack_name} attack and {num_modify} modifications")

                p_vals = [test_watermark(
                    prompts, num_tokens, llm_config, method, method, f"{attack_name}_{num_modify}", seed=seed
                ) for seed in wm_seeds]
                tpr, _ = compute_tpr(p_vals, pvals_unwm[(method_name, key_name)], fpr)
                tprs[(method_name, key_name)].append(tpr)

    save_filename = f"Figures/tpr_vs_{attack_name}_attack_k{k}_b{b}.pdf"
    if attack_name=="duplicate":
        xlabel="Number of Related Word Insertions"
    elif attack_name=="modify":
        xlabel="Number of Unrelated Token Replacements"
    elif attack_name=="translate":
        xlabel="Number of Translated Token Replacements"
    elif attack_name=="mask":
        xlabel="Number of Masked Token Replacements"
    else:
        xlabel="Number of Word Modifications"
    # Generate plot
    plot_tpr_modifications(modifications, tprs, save_filename, xlabel, fpr)