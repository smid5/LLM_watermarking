from simmark.experiments.plot_p_value_dist import plot_p_value_dist
from simmark.experiments.k_and_b_heatmaps_pvalue import generate_p_value_heatmaps
from simmark.experiments.num_modifications_vs_p_value import generate_p_value_modification_experiment, generate_p_value_translation_experiment, generate_p_value_mask_modification_experiment
from simmark.experiments.sentence_length_vs_p_value import generate_sentence_length_p_values
from simmark.experiments.three_attacks_simmark_vs_p_value import generate_simmark_modification_experiment
from simmark.experiments.test_expmin_nowater import generate_simmark_vs_expmin_p_values
from simmark.experiments.translation_p_value_dist import translation_p_value_violin, plot_p_value_dist_translation
from simmark.experiments.robustness_vs_distortion import generate_robustness_vs_distortion
from simmark.experiments.radar_plot import generate_radar_plot, generate_radar_comparison

filename = 'data/prompts.txt'

# Define the range of k and b values
# k_values = [5, 10, 15, 20]
# b_values = [8, 16, 32, 64]
# num_tokens = 100
# generate_p_value_heatmaps(k_values, b_values, num_tokens, filename=filename)

k_values = [3, 4, 5, 8, 12]
b_values = [4, 6, 8, 10, 12]
num_tokens = 100
generate_p_value_heatmaps(k_values, b_values, num_tokens, filename=filename)

k = 4
b = 4
num_modifications = 10
generate_p_value_modification_experiment(filename, k=k, b=b, num_modifications=num_modifications, num_tokens=num_tokens)
generate_p_value_mask_modification_experiment(filename, k=k, b=b, num_modifications=num_modifications, num_tokens=num_tokens)
generate_p_value_translation_experiment(filename, k=k, b=b, num_modifications=num_modifications, num_tokens=num_tokens)

generate_simmark_modification_experiment(filename, k=k, b=b, num_modifications=num_modifications, num_tokens=num_tokens)


length_variations = list(range(25, 105, 5))
num_modifications = 1
generate_sentence_length_p_values(filename, k=k, b=b, num_modifications=num_modifications, length_variations=length_variations)

generate_simmark_vs_expmin_p_values(filename, k=k, b=b, length_variations=length_variations)

translation_p_value_violin(k=k, b=b, num_tokens=100, filename=filename)
plot_p_value_dist_translation(f"simmarkemp_{k}_{b}", 100, filename)
plot_p_value_dist_translation(f"simmark_{k}_{b}", 100, filename) #this seems to be working the best
plot_p_value_dist_translation("expmin", 100, filename)
plot_p_value_dist_translation("expminnohash", 100, filename)
plot_p_value_dist_translation("softred", 100, filename)
plot_p_value_dist_translation("unigram", 100, filename)
plot_p_value_dist_translation("synthid", 100, filename)

generate_robustness_vs_distortion(filename, 100, k=k, b=b)
generate_robustness_vs_distortion(filename, 100, k=k, b=b, attack_name="translate")

generate_radar_plot(k=k, b=b, num_tokens=100, filename=filename, log=False)
generate_radar_plot(k=k, b=b, num_tokens=100, filename=filename, log=True)

kb_list = [(3,6), (4,4), (5,4), (5,6)]
generate_radar_comparison(kb_list, num_tokens=100, filename=filename)