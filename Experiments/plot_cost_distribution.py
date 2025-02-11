# plots the distribution of the cost of each generation type
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

import model_utils as utils
import attacks 

# stores sentence_starters.txt as variable
file_id = "1Bl1v09BK1TLX0RhE8YkFUGfmKD-7KUyN"
file_name = "sentence_starters.txt"
# !wget -O {file_name} "https://drive.google.com/uc?id={file_id}"

num_tokens = 100
k = 2
b = 64

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "facebook/opt-1.3b"
vocab_size = 50272
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def main():
    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # takes a file with sentence-starting phrases and generates text with them
    # compares the detection cost using simhash of text generated
    # without watermarking, with simhash, and with one word changed after simhash
    with open(file_name, 'r') as f: # text file generated using Chat-GPT
        cost_normal = []
        cost_simhash = []
        cost_modified_simhash = []

        # Iterate over each line in the file (one line at a time)
        for line in f:
            if line:
                line = line.strip()
                print(line)
                output_simhash, detect_simhash = utils.apply_watermarking(k, b, tokenizer, model, line, num_tokens, "simhash")

                cost_normal.append(utils.apply_watermarking(k, b, tokenizer, model, line, num_tokens, "unrelated")[1])
                cost_simhash.append(detect_simhash)

                modified_text = attacks.modify_text(tokenizer, vocab_size, output_simhash, 1)
                cost_modified_simhash.append(utils.apply_watermarking(k, b, tokenizer, model, modified_text, num_tokens, "attack_watermark"))

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

    plt.savefig("../Figures/Distribution of SimHash cost.png")

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()

    