from simmark.experiments.plot_p_value_dist import plot_p_value_dist
from simmark.experiments.k_and_b_heatmaps_pvalue import generate_p_value_heatmaps
from simmark.experiments.num_modifications_vs_p_value import generate_p_value_modification_experiment
from simmark.experiments.sentence_length_vs_p_value import generate_sentence_length_p_values

filename = 'data/prompts.txt'

plot_p_value_dist(k=2, b=64, num_tokens=30, filename=filename)
plot_p_value_dist(k=4, b=64, num_tokens=30, filename=filename)
plot_p_value_dist(k=8, b=64, num_tokens=30, filename=filename)


# Define the range of k and b values
k_values = [5, 10, 15, 20]
b_values = [8, 16, 32, 64]
num_tokens = 100
generate_p_value_heatmaps(k_values, b_values, num_tokens, filename=filename)

k = 5
b = 16
num_modifications = 21
generate_p_value_modification_experiment(filename, k=k, b=b, num_modifications=num_modifications, num_tokens=num_tokens)


length_variations = list(range(25, 105, 5))
num_modifications = 1
generate_sentence_length_p_values(filename, k=k, b=b, num_modifications=num_modifications, length_variations=length_variations)