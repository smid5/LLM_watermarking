import os
import methods.method1 as method1
import methods.method2 as method2
import methods.method3 as method3

def load_text_data(file_path):
    """Loads text data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def apply_watermarking(text, method="method1"):
    """Applies a selected watermarking method to the given text."""
    if method == "method1":
        return method1.watermark(text)
    elif method == "method2":
        return method2.watermark(text)
    elif method == "method3":
        return method3.watermark(text)
    else:
        raise ValueError("Unknown method selected.")

def evaluate_results(original_text, watermarked_text):
    """Compares original vs. watermarked text using some metric."""
    # Example: Check similarity
    return sum(1 for a, b in zip(original_text, watermarked_text) if a == b) / len(original_text)

def save_results(results, filename="experiment_results.txt"):
    """Saves experiment results to a file."""
    with open(filename, "w") as f:
        f.write("\n".join(results))

---

# stores sentence_starters.txt as variable
file_id = "1Bl1v09BK1TLX0RhE8YkFUGfmKD-7KUyN"
file_name = "sentence_starters.txt"
!wget -O {file_name} "https://drive.google.com/uc?id={file_id}"

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
            output_normal = generate(vocab_size, 60, line)
            cost_normal.append(simhash_detect(vocab_size, output_normal))
            output_simhash = simhash_generate(vocab_size, 60, line)
            cost_simhash.append(simhash_detect(vocab_size, output_simhash))
            cost_modified_simhash.append(simhash_detect(vocab_size, modify_text(vocab_size, output_simhash, 1)))