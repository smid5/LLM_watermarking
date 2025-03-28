# plots the distribution of the cost of each generation type
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

from .utils import load_llm_config, test_watermark, load_prompts, cbcolors, linestyles

def plot_p_value_dist(k, b, num_tokens, filename):
    llm_config = load_llm_config('facebook/opt-125m')

    prompts = load_prompts(filename=filename)

    detection_name = f"simmark_{k}_{b}"

    p_values = {}

    # Generate without watermark, no attack, and detection
    p_values['No Watermark'] = test_watermark(
        prompts, num_tokens, llm_config, "nomark", detection_name
    )

    # Generate with simhash watermark, no attack, and detection
    p_values['SimMark'] = test_watermark(
        prompts, num_tokens, llm_config, f"simmark_{k}_{b}", detection_name
    )

    # Generate with simhash watermark, with attack, and detection
    p_values['SimMark + Attack 1'] = test_watermark(
        prompts, num_tokens, llm_config, f"simmark_{k}_{b}", detection_name, "modify_1"
    )

    plt.style.use(['science'])
    plt.figure(figsize=(4, 3))

    # Labels and legend
    plt.xscale("log")
    for idx, key in enumerate(p_values):
        sns.kdeplot(p_values[key], label=key, log_scale=True, linewidth=2, color=cbcolors[idx], linestyle=linestyles[idx], cut=0)
    
    plt.xlabel(r"$p$-value")
    plt.ylabel("Frequency")
    plt.title(rf"SimMark $p$-values for $k={k}$, $b={b}$, $n={num_tokens}$")
    plt.legend()

    plt.savefig(f"figures/p_val_dist_simmark_{k}_{b}_{num_tokens}.pdf")

    # Show the plot
