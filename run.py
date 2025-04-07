from simmark.experiments.plot_p_value_dist import plot_p_value_dist
from simmark.experiments.k_and_b_heatmaps_pvalue import generate_p_value_heatmaps
from simmark.experiments.num_modifications_vs_p_value import generate_p_value_modification_experiment
from simmark.experiments.sentence_length_vs_p_value import generate_sentence_length_p_values
from simmark.experiments.three_attacks_simmark_vs_p_value import generate_simmark_modification_experiment
from simmark.experiments.test_expmin_nowater import generate_simmark_vs_expmin_p_values
from simmark.experiments.translation_p_value_dist import translation_p_value_violin, plot_p_value_dist_translation
from simmark.experiments.robustness_vs_distortion import generate_robustness_vs_distortion

filename = 'data/prompts.txt'
# plot_p_value_dist(k=2, b=64, num_tokens=30, filename=filename)
# plot_p_value_dist(k=4, b=64, num_tokens=30, filename=filename)
# plot_p_value_dist(k=8, b=64, num_tokens=30, filename=filename)

# # Define the range of k and b values
# k_values = [5, 10, 15, 20]
# b_values = [8, 16, 32, 64]
# num_tokens = 100
# generate_p_value_heatmaps(k_values, b_values, num_tokens, filename=filename)

# k = 5
# b = 16
# num_modifications = 10
# generate_p_value_modification_experiment(filename, k=k, b=b, num_modifications=num_modifications, num_tokens=num_tokens)

# generate_simmark_modification_experiment(filename, k=k, b=b, num_modifications=num_modifications, num_tokens=num_tokens)


# length_variations = list(range(25, 105, 5))
# num_modifications = 1
# generate_sentence_length_p_values(filename, k=k, b=b, num_modifications=num_modifications, length_variations=length_variations)

# generate_simmark_vs_expmin_p_values(filename, k=k, b=b, length_variations=length_variations)

# translation_p_value_violin(k=5, b=8, num_tokens=50, filename=filename)

# plot_p_value_dist_translation("simmark_5_8", 50, filename) 
# plot_p_value_dist_translation("simmark_5_8", 100, filename) #these seem to be working the best
plot_p_value_dist_translation("simmark_5_2", 100, filename)
plot_p_value_dist_translation("simmark_5_3", 100, filename)
plot_p_value_dist_translation("simmark_5_4", 100, filename)
plot_p_value_dist_translation("simmark_5_5", 100, filename)

# plot_p_value_dist_translation("expminnohash", 100, filename) 
# plot_p_value_dist_translation("expmin", 100, filename) 
# plot_p_value_dist_translation("simmark_5_8", 100, filename) 
# plot_p_value_dist_translation("unigram", 100, filename) 
# generate_robustness_vs_distortion(filename, 100, k=5, b=8)
generate_robustness_vs_distortion(filename, 100, k=5, b=2)
generate_robustness_vs_distortion(filename, 100, k=5, b=3)
generate_robustness_vs_distortion(filename, 100, k=5, b=4)
generate_robustness_vs_distortion(filename, 100, k=5, b=5)