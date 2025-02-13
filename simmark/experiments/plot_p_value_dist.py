# plots the distribution of the cost of each generation type
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import load_llm_config, test_watermark

num_tokens = 10
k = 2
b = 64

def plot_p_value_dist(filename):
    llm_config = load_llm_config('facebook/opt-125m')
    # takes a file with sentence-starting phrases and generates text with them
    # compares the detection cost using simhash of text generated
    # without watermarking, with simhash, and with one word changed after simhash
    # Load all prompts from file
    prompts = []
    with open(filename, 'r') as f: 
        for line in f:
            prompts.append(line.strip())
    
    detection_name = f"simmark_{k}_{b}"

    # Generate without watermark, no attack, and detection
    p_values_nowatermark = test_watermark(
        prompts, num_tokens, llm_config, "nomark", detection_name
    )

    # Generate with simhash watermark, no attack, and detection
    p_values_simmark = test_watermark(
        prompts, num_tokens, llm_config, f"simmark_{k}_{b}", detection_name
    )

    attack_name = "modify_1"

    # Generate with simhash watermark, with attack, and detection
    p_values_modified_simmark = test_watermark(
        prompts, num_tokens, llm_config, f"simmark_{k}_{b}", detection_name, attack_name
    )

        
    # Plot KDE curves
    plt.figure(figsize=(8, 6))
    sns.kdeplot(p_values_nowatermark, label="No Watermarking", linewidth=2)
    sns.kdeplot(p_values_simmark, label="SimMark", linewidth=2)
    sns.kdeplot(p_values_modified_simmark, label="SimMark with 1 word change", linewidth=2)

    # Labels and legend
    plt.xlabel(r"$p$-value")
    plt.ylabel("Frequency")
    plt.title(r"Distribution $p$-values of Detection Cost")
    plt.legend()
    plt.grid()

    plt.savefig("figures/p_val_dist.png")

    # Show the plot
    plt.show()    