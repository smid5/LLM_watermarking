import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, modify_text, delete_text, insert_text, cbcolors, linestyles

def plot_p_value_modifications(modifications, p_values_dict, filename):
    plt.style.use(['science'])
    plt.figure(figsize=(6, 4))

    # Iterate through each modification type and plot its data
    for idx, (label, values) in enumerate(p_values_dict.items()):
        plt.plot(modifications, values, marker='o', linestyle='-', color=cbcolors[idx], linewidth=2, label=label)

    plt.yscale("log")  # Set y-axis to log scale for better visualization
    plt.xlabel("Number of Modifications")
    plt.ylabel("Median p-Value")
    plt.title("Effect of Different Modification Types on SimMark p-Values")
    plt.legend()
    plt.grid()
    plt.xticks(modifications)
    plt.savefig(filename)
    plt.close()

def generate_simmark_modification_experiment(filename, k=2, b=64, num_modifications=21, num_tokens=100):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    modifications = np.arange(num_modifications)

    # Dictionary to store p-values for each modification type
    p_values = {
        "SimMark - Substitution": np.zeros(num_modifications),
        "SimMark - Insertion": np.zeros(num_modifications),
        "SimMark - Deletion": np.zeros(num_modifications)
    }
    
    for prompt in prompts:
        print(f"Processing: {prompt}")

        for i in range(num_modifications):
            # Apply different modification types
            output_substitution = modify_text(llm_config['tokenizer'], llm_config['vocab_size'], prompt, i)
            output_insertion = insert_text(llm_config['tokenizer'], llm_config['vocab_size'], prompt, i)
            output_deletion = delete_text(llm_config['tokenizer'], prompt, i)

            # Compute p-values for each modification type
            p_values["SimMark - Substitution"][i] += test_watermark(
                [output_substitution], num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}", f"modify_{i}"
            )[0]

            p_values["SimMark - Insertion"][i] += test_watermark(
                [output_insertion], num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}", f"insert_{i}"
            )[0]

            p_values["SimMark - Deletion"][i] += test_watermark(
                [output_deletion], num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}", f"delete_{i}"
            )[0]
    
    # Compute the average p-values across prompts
    for key in p_values:
        p_values[key] /= len(prompts)

    # Generate plot
    plot_p_value_modifications(modifications, p_values, f"figures/simmark_p_value_vs_modifications_k{k}_b{b}.pdf")

if __name__ == '__main__':
    generate_simmark_modification_experiment("sentence_starters.txt")
