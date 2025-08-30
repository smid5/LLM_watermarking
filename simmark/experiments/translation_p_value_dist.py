# plots the distribution of the cost of each generation type and its translation-attacked text
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
import numpy as np

from .utils import load_llm_config, test_watermark, load_prompts, cbcolors, linestyles, METHODS

def translation_p_value_violin(filename, k=4, b=4, num_tokens=100):
    llm_config = load_llm_config('facebook/opt-125m')

    prompts = load_prompts(filename=filename)

    p_values = {"SimMark": {}, "SoftRedList": {}, "Unigram": {}, "ExpMin": {}, "SynthID": {}}
    gen_methods = [f"simmark_{k}_{b}", "softred", "unigram", "expmin", "synthid"]

    for method, gen_method in zip(list(p_values.keys()), gen_methods):
        # Generate without watermark, no attack, and detection
        p_values[method]['No Watermark'] = test_watermark(
            prompts, num_tokens, llm_config, "nomark", gen_method
        )

        # Generate with simhash watermark, no attack, and detection
        p_values[method]['Watermark'] = test_watermark(
            prompts, num_tokens, llm_config, gen_method, gen_method
        )

        # Generate with simhash watermark, with attack, and detection
        p_values[method]['Watermark + Translate'] = test_watermark(
            prompts, num_tokens, llm_config, gen_method, gen_method, "translate"
        )

    # Flatten dictionary into a list of rows
    data = []
    for technique, categories in p_values.items():
        for category, p_values in categories.items():
            for p in p_values:
                data.append([technique, category, p])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Watermark", "Category", "p-value"])
    
    # Plot
    plt.style.use(['science'])
    plt.figure(figsize=(4, 3))
    sns.violinplot(
        data=df,
        x="Watermark",
        y="p-value",
        hue="Category",
        split=False,  
        palette={"No Watermark": "blue", "Watermark": "green", "Watermark + Translate": "red"},
        density_norm="count",
        inner="quart",
        cut=0
    )

    plt.xticks(rotation=20, ha="right", fontsize=12)
    plt.yscale("log")  # Log scale for p-values
    plt.ylim(1e-20, 1)
    plt.ylabel("p-value (log scale)")
    plt.xlabel("Watermarking Technique")
    # plt.title(rf"Distribution of p-values for all methods")
    plt.legend(title="Category", loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(f"Figures/translation_p_val_dist_{k}_{b}_{num_tokens}.pdf")

def plot_p_value_dist_translation(method_name, num_tokens, filename, k=4, b=4):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)

    method = f"simmark_{k}_{b}" if method_name == "SimMark" else METHODS[method_name]

    detection_name = method

    p_values = {}

    # Generate without watermark, no attack, and detection
    p_values['No Watermark'] = test_watermark(
        prompts, num_tokens, llm_config, "nomark", detection_name
    )

    # Generate with simhash watermark, no attack, and detection
    p_values[method_name] = test_watermark(
        prompts, num_tokens, llm_config, method, detection_name
    )

    # Generate with watermark, with attack, and detection
    p_values[f'{method_name} + Translation'] = test_watermark(
        prompts, num_tokens, llm_config, method, detection_name, "translate"
    )

    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(4, 3))

    # Labels and legend
    plt.xscale("log")

    threshold = 1e-40  # Exclude values below this
    for idx, key in enumerate(p_values):
        filtered_p_values = np.array(p_values[key])
        filtered_p_values = filtered_p_values[filtered_p_values > threshold]  # Remove small values

        if len(filtered_p_values) > 0:
            sns.kdeplot(filtered_p_values, 
                        label=key, 
                        log_scale=True, 
                        linewidth=2, 
                        color=cbcolors[idx], 
                        linestyle=linestyles[idx],
                        cut=0)
    
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig(f"Figures/translation_p_val_dist_{method}_{num_tokens}.pdf")