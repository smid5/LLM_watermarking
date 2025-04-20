import os
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, cbcolors, linestyles

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def plot_sentence_length_p_values(sentence_lengths, p_values, filename):
    plt.style.use(['science'])
    plt.figure(figsize=(6, 4))
    
    for idx, (label, values) in enumerate(p_values.items()):
        plt.plot(sentence_lengths, values, marker='o', linestyle='-', color=cbcolors[idx], linewidth=2, label=label)
    
    plt.yscale("log")  # Set y-axis to log scale for better visualization
    plt.xscale("log")
    plt.xlabel("Sentence Length")
    plt.ylabel("Mean p-Value")
    plt.title("SimMark vs ExpMin (Non-Watermarked) Sentence Length vs. Mean p-Value")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

def generate_simmark_vs_expmin_p_values(filename, k=2, b=64, length_variations=list(range(25, 105, 5))):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    
    p_values = {"SimMark": {}, "ExpMin on No Watermark": {}}
    
    for length in length_variations:
        applicable_prompts = [p for p in prompts if len(p.split()) < length]
        if not applicable_prompts:
            continue
        
        num_tokens_list = [length - len(p.split()) for p in applicable_prompts]
        
        # SimMark Detection on SimMark-Generated Text
        p_values["SimMark"][length] = np.mean(
            [test_watermark([p], num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )
        
        # ExpMin Detection on Non-Watermarked Text
        p_values["ExpMin on No Watermark"][length] = np.mean(
            [test_watermark([p], num_tokens, llm_config, "nomark", "expmin")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )
    
    sorted_lengths = sorted(p_values["SimMark"].keys())
    for key in p_values:
        p_values[key] = [p_values[key][l] for l in sorted_lengths]
    
    plot_sentence_length_p_values(sorted_lengths, p_values, f"figures/sentence_length_vs_p_values_simmark_vs_expmin_k{k}_b{b}.pdf")