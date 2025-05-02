import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import load_llm_config, test_watermark, load_prompts
import matplotlib.colors as mcolors

def plot_heatmap(data, k_values, b_values, title, filename):
    plt.figure(figsize=(6, 5))

    # Avoid log(0) errors by setting a small lower bound
    data = np.maximum(data, 1e-20)

    sns.heatmap(data, annot=True, fmt=".1e", cmap="coolwarm", norm=mcolors.LogNorm(vmin=data.min(), vmax=data.max()), 
                xticklabels=b_values, yticklabels=k_values)

    # sns.heatmap(data, annot=True, fmt=".1e", cmap="coolwarm", xticklabels=b_values, yticklabels=k_values)
    plt.xlabel("b")
    plt.ylabel("k")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def generate_p_value_heatmaps(k_values, b_values, num_tokens, filename):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)
    
    p_values_watermark = np.zeros((len(k_values), len(b_values)))
    p_values_unrelated = np.zeros((len(k_values), len(b_values)))
    p_values_modified = np.zeros((len(k_values), len(b_values)))
    p_values_translated = np.zeros((len(k_values), len(b_values)))
    
    for i, k in enumerate(k_values):
        for j, b in enumerate(b_values):
            # print("k: " + str(k))
            # print("b: " + str(b))
            detection_name = f"simmark_{k}_{b}"
            
            # print("watermark")
            p_values_watermark[i, j] = np.median(test_watermark(
                prompts, num_tokens, llm_config, f"simmark_{k}_{b}", detection_name
            ))
            
            # print("nomark")
            p_values_unrelated[i, j] = np.median(test_watermark(
                prompts, num_tokens, llm_config, "nomark", detection_name
            ))
            
            # print("modify")
            p_values_modified[i, j] = np.median(test_watermark(
                prompts, num_tokens, llm_config, f"simmark_{k}_{b}", detection_name, "modify_1"
            ))

            p_values_translated[i, j] = np.median(test_watermark(
                prompts, num_tokens, llm_config, f"simmark_{k}_{b}", detection_name, "translate"
            ))
    
    plot_heatmap(p_values_watermark, k_values, b_values, "p-Values for Watermarked Text", "figures/heatmap_watermark.pdf")
    plot_heatmap(p_values_unrelated, k_values, b_values, "p-Values for Unrelated Text", "figures/heatmap_unrelated.pdf")
    plot_heatmap(p_values_modified, k_values, b_values, "p-Values for Modified Text", "figures/heatmap_modified.pdf")
    plot_heatmap(p_values_translated, k_values, b_values, "p-Values for Translated Text", "figures/heatmap_translated.pdf")