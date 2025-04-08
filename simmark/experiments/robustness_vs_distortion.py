import numpy as np
import matplotlib.pyplot as plt

from .utils import load_llm_config, load_prompts, test_watermark, test_distortion

# def plot_robustness_vs_distortion(robustness, distortion, filename, k, b):
#     methods = list(robustness.keys())
#     robustness_values = list(robustness.values())
#     distortion_values = list(distortion.values())
#     detections = ["Watermarked", "Watermarked+Translated", "Unwatermarked"]
#     markers = ['o', 's', '^'] 

#     plt.figure(figsize=(8, 6))
#     for i, key in enumerate(methods):
#         for j, detection in enumerate(detections):
#             plt.scatter(robustness_values[i][j], distortion_values[i], marker=markers[j], label=key)

#     # Labels and legend
#     plt.xlabel('Robustness (average p-value on translated text)')
#     plt.ylabel('Distortion (perplexity)')
#     plt.title(f'Robustness vs. Distortion for k={k}, b={b}')
#     plt.legend()

#     # Show plot
#     plt.grid(True)
#     plt.savefig(filename)
#     plt.close()

def plot_robustness_vs_distortion(robustness, distortion, filename, k, b, num_tokens):
    methods = list(robustness.keys())
    robustness_values = list(robustness.values())
    distortion_values = list(distortion.values())
    detections = ["Watermarked", "Watermarked+Translated", "Unwatermarked"]
    markers = ['o', 's', '^']
    colors = plt.cm.tab10.colors  # Assign distinct colors

    plt.figure(figsize=(8, 6))

    # Store legend handles
    technique_handles = []
    detection_handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='black', markersize=8, label=d) for m, d in zip(markers, detections)]

    # Scatter plot with correct labeling
    for i, key in enumerate(methods):
        xs = robustness_values[i]
        ys = distortion_values[i]

        # Plot line connecting the points
        plt.plot(xs, ys, color=colors[i % len(colors)], linewidth=1)

        for j, detection in enumerate(detections):
            plt.scatter(xs[j], ys[j], 
                        color=colors[i % len(colors)], marker=markers[j])
        technique_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i % len(colors)], markersize=8, label=key))

    # Labels and legends
    plt.xlabel('Robustness (median p-value)')
    plt.ylabel('Distortion (median perplexity)')
    plt.title(f'Robustness vs. Distortion for k={k}, b={b}, n={num_tokens}')
    
    # Two separate legends
    legend1 = plt.legend(handles=technique_handles, title="Watermarking Technique", loc='upper right', bbox_to_anchor=(1.3, 0.4))
    legend2 = plt.legend(handles=detection_handles, title="Detection Type", loc='upper right', bbox_to_anchor=(1.3, 0))
    
    plt.gca().add_artist(legend1)  # Ensure first legend is not overridden

    # Show and save
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def generate_robustness_vs_distortion(filename, num_tokens, k=5, b=8):
    llm_config = load_llm_config('facebook/opt-125m')
    prompts = load_prompts(filename=filename)

    robustness = {}
    distortion = {}
    generation_methods = [f"simmark_{k}_{b}", "unigram", "softred", "expmin", "synthid", "expminnohash", "nomark"]
    detection_methods = [f"simmark_{k}_{b}", "unigram", "softred", "expmin", "synthid", "expminnohash", f"simmark_{k}_{b}"]

    for generation_method, detection_method in zip(generation_methods, detection_methods):
        robustness[generation_method] = [np.median(test_watermark(prompts, num_tokens, llm_config, generation_method, detection_method)),
                                         np.median(test_watermark(prompts, num_tokens, llm_config, generation_method, detection_method, "translate")),
                                         np.median(test_watermark(prompts, num_tokens, llm_config, "nomark", detection_method))]
        distortion[generation_method] = [np.median(test_distortion(prompts, num_tokens, llm_config, generation_method, detection_method)),
                                         np.median(test_distortion(prompts, num_tokens, llm_config, generation_method, detection_method, "translate")),
                                         np.median(test_distortion(prompts, num_tokens, llm_config, "nomark", detection_method))]

    plot_robustness_vs_distortion(robustness, distortion, f"Figures/robustness_distortion_k{k}_b{b}_{num_tokens}.pdf", k, b, num_tokens)
