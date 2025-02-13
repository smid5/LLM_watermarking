# plots the distribution of the cost of each generation type
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

from .utils import load_llm_config, test_watermark

num_tokens = 30
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

    # Plot the distribution of p-values
    sns.histplot(p_values_nowatermark, label="No watermark", color="blue", kde=True)
    sns.histplot(p_values_simmark, label="SimHash watermark", color="orange", kde=True)
    sns.histplot(p_values_modified_simmark, label="SimHash watermark, modified", color="green", kde=True)


    # Labels and legend
    plt.xlabel(r"$p$-value")
    plt.ylabel("Frequency")
    plt.title(r"Distribution $p$-values of Detection Cost")
    plt.legend()
    plt.grid()

    plt.savefig(f"figures/p_val_dist_simmark_{k}_{b}.pdf")

    # Show the plot
