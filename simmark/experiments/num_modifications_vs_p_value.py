import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, modify_text, cbcolors, linestyles

def plot_p_value_modifications(modifications, average_p_values, filename):
    plt.style.use(['science'])
    plt.figure(figsize=(6, 4))
    plt.plot(modifications, average_p_values, marker='o', linestyle='-', color=cbcolors[0], linewidth=2, label="SimHash Generated Text")
    
    plt.yscale("log")  # Set y-axis to log scale to better capture small variations
    plt.xlabel("Number of Modifications")
    plt.ylabel("Average SimHash p-Value")
    plt.title("Effect of Modifications on SimHash p-Values")
    plt.legend()
    plt.grid()
    plt.xticks(modifications)
    plt.savefig(filename)
    plt.close()

def generate_p_value_modification_experiment(filename, k=2, b=64, num_modifications=21, num_tokens=100):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    modifications = np.arange(num_modifications)
    average_p_values = np.zeros(num_modifications)
    
    for prompt in prompts:
        print(f"Processing: {prompt}")
        
        for i in range(num_modifications):
            output_modified = modify_text(llm_config['tokenizer'], llm_config['vocab_size'], prompt, i)
            p_value = test_watermark([output_modified], num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}", f"modify_{i}")[0]
            average_p_values[i] += p_value
    
    average_p_values /= len(prompts)
    plot_p_value_modifications(modifications, average_p_values, f"figures/p_value_vs_modifications_k{k}_b{b}.pdf")

