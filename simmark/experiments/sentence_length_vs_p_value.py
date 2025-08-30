import os
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False

from .utils import load_llm_config, test_watermark, load_prompts, COLORS

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def plot_sentence_length_p_values(sentence_lengths, p_values, filename):
    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(7.5, 4))
    MIN_PVAL = 1e-16
    
    for idx, (label, values) in enumerate(p_values.items()):
        clipped_values = [max(v, MIN_PVAL) for v in values]
        plt.plot(sentence_lengths, clipped_values, marker='o', linestyle='-', color=COLORS[label], linewidth=2, label=label)
    
    plt.yscale("log")  # Set y-axis to log scale to better capture small variations
    plt.xscale("linear")
    plt.xlabel("Sentence Length")
    plt.ylabel("Median p-Value")

    yticks = [1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, MIN_PVAL]
    plt.yticks(yticks)

    # Format ticks: show "<10^-15" at bottom
    def custom_y_ticks(val, _):
        if val <= MIN_PVAL + 1e-20:  # Allow for float imprecision
            return r"$<10^{-16}$"
        exponent = int(np.log10(val))
        return f"$10^{{{exponent}}}$"

    plt.gca().yaxis.set_major_formatter(FuncFormatter(custom_y_ticks))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_sentence_length_p_values(filename, k=4, b=4, length_variations=list(range(25, 105, 5)), seeds=[42]):
    llm_config = load_llm_config("meta-llama/Llama-3.2-3B")
    prompts = load_prompts(filename=filename)
    
    p_values = {"No Watermark": {}, "SimMark": {}, "SoftRedList": {}, "Unigram": {}, "ExpMin": {}, "SynthID": {}}
    generation_methods = ["nomark", f"simmark_{k}_{b}", "softred", "unigram", "expmin", "synthid"]
    detection_methods = [f"simmark_{k}_{b}", f"simmark_{k}_{b}", "softred", "unigram", "expmin", "synthid"]
    
    for length in length_variations:
        applicable_prompts = [p for p in prompts if len(p.split()) < length]
        if not applicable_prompts:
            continue
        
        num_tokens_list = [length - len(p.split()) for p in applicable_prompts]

        for method_name, gen_method, det_method in zip(list(p_values.keys()), generation_methods, detection_methods):
            all_pvals = []
            for seed in seeds:
                new_data = [
                    test_watermark([p], num_tokens, llm_config, gen_method, det_method, seed=seed)[0]
                    for p, num_tokens in zip(applicable_prompts, num_tokens_list)
                ]
                all_pvals.extend(new_data)

            p_values[method_name][length] = np.median(all_pvals)

    sorted_lengths = sorted(p_values["No Watermark"].keys())
    for key in p_values:
        p_values[key] = [p_values[key][l] for l in sorted_lengths]
    
    plot_sentence_length_p_values(sorted_lengths, p_values, f"Figures/sentence_length_vs_p_values_k{k}_b{b}.pdf")

if __name__ == '__main__':
    generate_sentence_length_p_values("sentence_starters.txt")