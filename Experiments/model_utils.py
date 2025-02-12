import sys

sys.path.insert(0, '/Users/hayadiwan/Desktop/Tiny Projects/watermark-main/LLM_watermarking')

# from Unrelated.generate_unrelated import generate

# from Exp.generate_exp import exp_generate
# from Exp.detect_exp import exp_detect

# from Red_green.generate_red_green import red_green_generate
# from Red_green.detect_red_green import red_green_detect

# from Simhash.generate_simhash import simhash_generate
# from Simhash.detect_simhash import simhash_detect


from Methods.Unrelated.generate_unrelated import generate

# from Methods.Exp.generate_exp import exp_generate
# from Methods.Exp.detect_exp import exp_detect

# from Methods.Red_green.generate_red_green import red_green_generate
# from Methods.Red_green.detect_red_green import red_green_detect

from Methods.Simhash.generate_simhash import simhash_generate
from Methods.Simhash.detect_simhash import simhash_detect

vocab_size = 50272

# ============================================================
# Watermarking Utility Functions
# ============================================================

def apply_watermarking(k, b, tokenizer, model, text, num_tokens, method="exp"):
    detected_result = "" # REMOVE LATER

    """Applies a selected watermarking method to the given text."""
    if method == "exp": # TO DO
        # generated_text = method1.generate(text)
        # detected_result = method1.detect(generated_text)
        print("TO DO exp")

        return detected_result
    elif method == "simhash": # DONE
        output_simhash = simhash_generate(tokenizer, model, vocab_size, num_tokens, text, k, b)
        _, detect_simhash = simhash_detect(tokenizer, model, vocab_size, output_simhash, k, b)
        return output_simhash, detect_simhash
    elif method == "red_green": # TO DO
        print("TO DO red_green")
        # generated_text = method3.generate(text)
        # detected_result = method3.detect(generated_text)
        return detected_result
    elif method == "unrelated": # DONE
        output_normal = generate(tokenizer, model, vocab_size, num_tokens, text)
        _, detect_normal = simhash_detect(tokenizer, model, vocab_size, output_normal, k, b)
        return output_normal, detect_normal
    elif method == "attack_watermark":
        _, detect_modified = simhash_detect(tokenizer, model, vocab_size, text, k, b)
        return detect_modified
    else:
        raise ValueError(f"Unknown method selected: {method}")







