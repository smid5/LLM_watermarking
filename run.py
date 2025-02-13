from simmark.experiments.plot_p_value_dist import plot_p_value_dist

filename = 'data/prompts.txt'
plot_p_value_dist(k=2, b=64, num_tokens=30, filename=filename)
plot_p_value_dist(k=4, b=64, num_tokens=30, filename=filename)
plot_p_value_dist(k=8, b=64, num_tokens=30, filename=filename)


