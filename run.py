from simmark.experiments.plot_p_value_dist import plot_p_value_dist
from simmark.experiments.k_and_b_heatmaps_pvalue import generate_p_value_heatmaps
from simmark.experiments.num_modifications_vs_p_value import plot_modification_comparison
from simmark.experiments.sentence_length_vs_p_value import generate_sentence_length_p_values
from simmark.experiments.translation_p_value_dist import translation_p_value_violin, plot_p_value_dist_translation
from simmark.experiments.robustness_vs_distortion import generate_robustness_vs_distortion
from simmark.experiments.radar_plot import generate_radar_plot
from simmark.experiments.plot_distortion_dist import plot_distortion_dist

filename = 'data/prompts.txt'
num_tokens = 100

# # Define the range of k and b values
# k_values = [5, 10, 15, 20]
# b_values = [8, 16, 32, 64]
# generate_p_value_heatmaps(k_values, b_values, num_tokens, filename=filename)

k_values = [3, 4, 5, 8, 12]
b_values = [4, 6, 8, 10, 12]
generate_p_value_heatmaps(k_values, b_values, num_tokens, filename=filename)

length_variations = list(range(25, 105, 5))
generate_sentence_length_p_values(filename, length_variations=length_variations)

modification_values = list(range(0, 31, 3))
attacks = ["modify", "translate", "duplicate"]
plot_modification_comparison(filename=filename, attacks=attacks, modification_values=modification_values, num_tokens=num_tokens)

translation_p_value_violin(filename=filename, num_tokens=num_tokens)
plot_p_value_dist_translation("SimMark", num_tokens, filename)
plot_p_value_dist_translation("ExpMin", num_tokens, filename)
plot_p_value_dist_translation("SoftRedList", num_tokens, filename)
plot_p_value_dist_translation("Unigram", num_tokens, filename)
plot_p_value_dist_translation("SynthID", num_tokens, filename)
plot_p_value_dist("SimMark", num_tokens, filename)

plot_distortion_dist(num_tokens, filename)

generate_radar_plot(num_tokens=num_tokens, filename=filename)

generate_robustness_vs_distortion(filename, num_tokens)
generate_robustness_vs_distortion(filename, num_tokens, attack_name="translate")