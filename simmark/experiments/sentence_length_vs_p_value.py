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
    
    plt.yscale("log")  # Set y-axis to log scale to better capture small variations
    plt.xscale("log")
    plt.xlabel("Sentence Length")
    plt.ylabel("Median p-Value")
    plt.title("Sentence Length vs. Median p-Value")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()

def generate_sentence_length_p_values(filename, k=2, b=64, num_modifications=1, length_variations=list(range(25, 105, 5))):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    
    p_values = {"No Watermark": {}, "SimMark": {}, "SimMark + Attack": {}, "SimMark Empirical": {}, "SoftRedList": {}, "Unigram": {}, "ExpMin": {}, "ExpMinNoHash": {}, "SynthID": {}}
    
    for length in length_variations:
        applicable_prompts = [p for p in prompts if len(p.split()) < length]
        if not applicable_prompts:
            continue
        
        num_tokens_list = [length - len(p.split()) for p in applicable_prompts]
        
        p_values["No Watermark"][length] = np.median(
            [test_watermark([p], num_tokens, llm_config, "nomark", f"simmark_{k}_{b}")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )
        
        p_values["SimMark"][length] = np.median(
            [test_watermark([p], num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )
        
        p_values["SimMark + Attack"][length] = np.median(
            [test_watermark([p], num_tokens, llm_config, f"simmark_{k}_{b}", f"simmark_{k}_{b}", f"modify_{num_modifications}")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )

        p_values["SimMark Empirical"][length] = np.median(
            [test_watermark([p], num_tokens, llm_config, f"simmarkemp_{k}_{b}", f"simmarkemp_{k}_{b}")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )
        
        p_values["SoftRedList"][length] = np.median(
            [test_watermark([p], num_tokens, llm_config, "softred", "softred")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )

        p_values["Unigram"][length] = np.median(
            [test_watermark([p], num_tokens, llm_config, "unigram", "unigram")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )
        
        p_values["ExpMin"][length] = np.median(
            [test_watermark([p], num_tokens, llm_config, "expmin", "expmin")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )

        p_values["ExpMinNoHash"][length] = np.median(
            [test_watermark([p], num_tokens, llm_config, "expminnohash", "expminnohash")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )

        p_values["SynthID"][length] = np.median(
            [test_watermark([p], num_tokens, llm_config, "synthid", "synthid")[0] 
             for p, num_tokens in zip(applicable_prompts, num_tokens_list)]
        )
    
    sorted_lengths = sorted(p_values["No Watermark"].keys())
    for key in p_values:
        p_values[key] = [p_values[key][l] for l in sorted_lengths]
    
    plot_sentence_length_p_values(sorted_lengths, p_values, f"figures/sentence_length_vs_p_values_k{k}_b{b}.pdf")

if __name__ == '__main__':
    generate_sentence_length_p_values("sentence_starters.txt")