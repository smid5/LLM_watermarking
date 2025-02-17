import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import numpy as np
import matplotlib.pyplot as plt

# import simmark.experiments.utils as utils
from ..experiments import utils
from ..experiments import attacks 

# stores sentence_starters.txt as variable
file_id = "1Bl1v09BK1TLX0RhE8YkFUGfmKD-7KUyN"
file_name = "sentence_starters.txt"

num_tokens = 100
k = 2
b = 64
vocab_size = 50272


def main():
    # incrementally modifies text generated using simHash and compares its cost to text generated without watermarking
    num_lines = 0
    num_modifications = 21
    average_cost_simhash = [0] * num_modifications
    average_cost_normal = 0

    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    with open(file_name, 'r') as f2:
        for line in f2:
            if line:
                num_lines += 1
                line = line.strip()
                print(line)
                output_simhash, _ = utils.apply_watermarking(k, b, tokenizer, model, line, num_tokens, "simhash")

                # output_simhash = simhash_generate(vocab_size, 60, line)
                for i in range(num_modifications):
                    # print("i: " + str(i))
                    output_modified = attacks.modify_text(tokenizer, vocab_size, output_simhash, i)
                    average_cost_simhash[i] += utils.apply_watermarking(k, b, tokenizer, model, output_modified, num_tokens, "attack_watermark")

                    # average_cost_simhash[i] += simhash_detect(vocab_size, output_modified)
                # output = generate(vocab_size, 60, line)
                average_cost_normal += utils.apply_watermarking(k, b, tokenizer, model, line, num_tokens, "unrelated")[1]
                # average_cost_normal += simhash_detect(vocab_size, output)

    for i in range(num_modifications):
        average_cost_simhash[i] /= num_lines
    average_cost_normal /= num_lines


    # plots the simHash cost of text without watermarking and text that is incrementally modified after simHash generation

    # X-axis values (modification indices)
    modifications = list(range(num_modifications))

    # Plot the line graph for average_cost_simhash
    plt.plot(modifications, average_cost_simhash, marker='o', linestyle='-', label="SimHash Generated Text", color='b')

    # Plot the horizontal line for average_cost_normal
    plt.axhline(y=average_cost_normal, color='r', linestyle='--', label="Text without Watermarking")

    # Labels and title
    plt.xlabel("Number of Modifications")
    plt.ylabel("Average SimHash Cost")
    plt.title("SimHash Cost Comparison of SimHash and Normal Generation")
    plt.legend()
    plt.grid()
    plt.xticks(modifications)

    plt.savefig("../Figures/SimHash Cost Comparison of SimHash and Normal Generation.png")

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()