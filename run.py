from simmark.experiments.plot_p_value_dist import plot_p_value_dist
from simmark.experiments.k_and_b_heatmaps_pvalue import generate_p_value_heatmaps
from simmark.experiments.num_modifications_vs_p_value import generate_p_value_modification_experiment
from simmark.experiments.sentence_length_vs_p_value import sentence_length_median_pvalue
from simmark.experiments.translation_p_value_dist import translation_p_value_violin, plot_p_value_dist_translation
from simmark.experiments.robustness_vs_distortion import generate_robustness_vs_distortion
from simmark.experiments.radar_plot import generate_radar_plot
from simmark.experiments.plot_distortion_dist import plot_distortion_dist
from simmark.experiments.plot_distortion_dist import sentence_length_median_distortion
from simmark.experiments.tpr import get_tprs
from simmark.experiments.num_modifications_vs_tpr import generate_tpr_modification_experiment

filename = 'data/prompts.txt'
num_tokens = 100
seeds = [42]
# model_name="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
model_name='meta-llama/Meta-Llama-3-8B'

# # Define the range of k and b values
# k_values = [5, 10, 15, 20]
# b_values = [8, 16, 32, 64]
# generate_p_value_heatmaps(k_values, b_values, num_tokens, filename=filename)

# k_values = [3, 4, 5, 8, 12]
# b_values = [4, 6, 8, 10, 12]
# generate_p_value_heatmaps(k_values, b_values, num_tokens, filename=filename)

# length_variations = list(range(25, 105, 5))
# generate_sentence_length_p_values(filename, length_variations=length_variations)

# modification_values = list(range(0, 31, 3))
# attacks = ["modify", "translate", "duplicate"]
# plot_modification_comparison(filename=filename, attacks=attacks, modification_values=modification_values, num_tokens=num_tokens)

# translation_p_value_violin(filename=filename, num_tokens=num_tokens)
# plot_p_value_dist_translation("SimMark", num_tokens, filename)
# plot_p_value_dist_translation("ExpMin", num_tokens, filename)
# plot_p_value_dist_translation("SoftRedList", num_tokens, filename)
# plot_p_value_dist_translation("Unigram", num_tokens, filename)
# plot_p_value_dist_translation("SynthID", num_tokens, filename)
# plot_p_value_dist("SimMark", num_tokens, filename)
# plot_p_value_dist_translation("expmin_simhash_4_4", num_tokens, filename, seeds=seeds)
# plot_p_value_dist_translation("expmin_standard_4_4", num_tokens, filename, seeds=seeds)
# plot_p_value_dist_translation("synthid_simhash_4_4", num_tokens, filename, seeds=seeds)
# plot_p_value_dist_translation("synthid_standard_4_4", num_tokens, filename, seeds=seeds)

# plot_distortion_dist(num_tokens, filename)

# generate_radar_plot(num_tokens=num_tokens, filename=filename, seeds=seeds)

# generate_robustness_vs_distortion(filename, num_tokens)
# generate_robustness_vs_distortion(filename, num_tokens, attack_name="translate")

# plot_p_value_dist_translation("WaterMax", "SimKey", num_tokens, filename, b=4, seeds=seeds, model_name=model_name)
# plot_p_value_dist_translation("WaterMax", "Standard", num_tokens, filename, b=4, seeds=seeds, model_name=model_name)
# plot_p_value_dist_translation("ExpMin", "SimKey", num_tokens, filename, seeds=seeds, model_name=model_name)
# plot_p_value_dist_translation("ExpMin", "Standard", num_tokens, filename, seeds=seeds, model_name=model_name)
# plot_p_value_dist_translation("SynthID", "SimKey", num_tokens, filename, seeds=seeds, model_name=model_name)
# plot_p_value_dist_translation("SynthID", "Standard", num_tokens, filename, seeds=seeds, model_name=model_name)

length_variations = list(range(25, 105, 5))
# sentence_length_median_distortion(length_variations, filename, seeds=seeds, model_name=model_name)
# sentence_length_median_pvalue(length_variations, filename, seeds=seeds, model_name=model_name)

method_names = ["ExpMin", "SynthID", "WaterMax"]
modification_values = list(range(15, 31, 15))
# generate_p_value_modification_experiment(modification_values, num_tokens, filename, attack_name="modify", method_names=method_names, seeds=seeds, model_name=model_name)
# generate_p_value_modification_experiment(modification_values, num_tokens, filename, attack_name="translate", method_names=method_names, seeds=seeds, model_name=model_name)
# generate_p_value_modification_experiment(modification_values, num_tokens, filename, attack_name="mask", seeds=seeds, model_name=model_name)

unwm_seeds = [42, 0, 1]
# generate_tpr_modification_experiment(modification_values, num_tokens, filename, attack_name="modify", method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name, output_log_file=True)
# generate_tpr_modification_experiment(modification_values, num_tokens, filename, attack_name="translate", method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name)
generate_tpr_modification_experiment(modification_values, num_tokens, filename, attack_name="mask", method_names=method_names, fpr=1e-2, unwm_seeds=unwm_seeds, wm_seeds=seeds, model_name=model_name)