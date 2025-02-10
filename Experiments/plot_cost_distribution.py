# plots the distribution of the cost of each generation type
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

# Plot KDE curves
plt.figure(figsize=(8, 6))
sns.kdeplot(cost_normal, label="No Watermarking", linewidth=2)
sns.kdeplot(cost_simhash, label="SimHash", linewidth=2)
sns.kdeplot(cost_modified_simhash, label="SimHash with 1 word change", linewidth=2)

# Labels and legend
plt.xlabel("Cost")
plt.ylabel("Frequency")
plt.title("Distribution of SimHash cost")
plt.legend()
plt.grid()

# Show the plot
plt.show()