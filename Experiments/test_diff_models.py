import subprocess
import os
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from EXP.transform.score import transform_score, transform_edit_score
from EXP.transform.sampler import transform_sampling
from EXP.transform.key import transform_key_func
from EXP.generation import generate, generate_rnd
from EXP.detection import phi,fast_permutation_test,permutation_test

def run_exp(prompt, model_name, vocab_size, n, m, k, seed):
    """
    Run generation and detection for the EXP method using the generate function.
    """
    print("Running EXP method...")

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")

    # Tokenize the prompt
    prompts = tokenizer([prompt], return_tensors="pt").input_ids

    # Define the generation function
    generate_watermark = lambda prompt, seed: generate(
        model=model,
        prompts=prompt,
        vocab_size=vocab_size,
        n=n,
        m=m,
        seeds=[seed],
        key_func=transform_key_func,
        sampler=transform_sampling,
        random_offset=True
    )

    # Generate tokens
    generated_tokens = generate_watermark(prompts, seed)

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("Generated Text (EXP Method):")
    print(generated_text)

    # Detection logic for EXP method
    if k > 0:
        dist = lambda x, y: transform_edit_score(x, y, gamma=0.0)
    else:
        dist = lambda x, y: transform_score(x, y)
    test_stat = lambda tokens, n, k, generator, vocab_size, null=False: phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=transform_key_func,
        vocab_size=vocab_size,
        dist=dist,
        null=null,
        normalize=True
    )

    # Run detection
    generator = torch.Generator()
    generator.manual_seed(seed)

    p_value = test_stat(
        tokens=generated_tokens.flatten(),
        n=n,
        k=k,
        generator=generator,
        vocab_size=vocab_size
    )

    print("Detection Results (EXP Method):")
    print(f"P-value: {p_value:.10f}")
    if p_value < 0.05:
        print("Watermark detected!")
    else:
        print("No watermark detected.")

    print("Finished running EXP method.")


def run_kgw(prompt, model_name, kgw_generate_file, kgw_detect_file, fraction, strength, wm_key, max_new_tokens, output_dir):
    """
    Run generation and detection for the KGW method.
    """
    print("Running KGW method...")

    # Step 1: Create a temporary prompt file
    temp_prompt_file = "temp_prompt.jsonl"
    with open(temp_prompt_file, "w") as f:
        f.write(f"{{\"prefix\": \"{prompt}\", \"gold_completion\": \"\"}}\n")

    # Step 2: Generate text using the KGW generation script
    generate_command = [
        "python3", kgw_generate_file,
        "--model_name", model_name,
        "--fraction", str(fraction),
        "--strength", str(strength),
        "--wm_key", str(wm_key),
        "--prompt_file", temp_prompt_file,
        "--output_dir", output_dir,
        "--max_new_tokens", str(max_new_tokens)
    ]

    # Force the device alignment
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # For Apple MPS, ensures fallback to CPU if needed

    subprocess.run(generate_command)

    # Retrieve the generated output
    output_file = os.path.join(output_dir, f"{model_name.replace('/', '-')}_strength_{strength}_frac_{fraction}_len_{max_new_tokens}_num_1.jsonl")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            output_data = [json.loads(line) for line in f]
            if output_data:
                generated_text = output_data[0].get("gen_completion", [""])[0]
                print("Generated Text (KGW Method):")
                print(generated_text)
            else:
                print("No generated text found in output file.")
    else:
        print("Output file not found.")

    # Step 3: Detect watermark using the KGW detection script
    # detect_command = [
    #     "python3", kgw_detect_file,
    #     "--input_file", output_file
    # ]
    # subprocess.run(detect_command)

    # # Clean up temporary file
    # os.remove(temp_prompt_file)
    print("Finished running KGW method.")


def run_custom(prompt, model_name, custom_generate_file, custom_detect_file, seeds, k, b, m, n, vocab_size):
    """
    Run generation and detection for the custom method.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Running custom method...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")  # Ensure model is accessible

    # Encode the prompt
    encoded_prompt = tokenizer(prompt, return_tensors="pt").input_ids

    # Step 1: Generate text using the custom generation script
    from generate_simhash import generate_with_simhash

    generated_tokens = generate_with_simhash(
        model=model,
        tokenizer=tokenizer,
        prompts=encoded_prompt,
        vocab_size=vocab_size,
        n=n,
        m=m,
        seeds=[seeds],
        k=k,
        b=b
    )

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Generated Text (Custom Method):")
    print(generated_text)

    # Step 2: Detect watermark using the custom detection script
    from detection_simhash import simhash_detect_with_permutation

    # Prepare the context and observed token for detection
    context = tokenizer.decode(generated_tokens[:-1], skip_special_tokens=True)
    observed_token = generated_tokens[-1]

    # Perform detection
    p_value, result, _ = simhash_detect_with_permutation(
        context=context,
        observed_token=observed_token,
        vocab_size=vocab_size,
        k=k,
        b=b,
        seed=seeds,
        model=model,
        tokenizer=tokenizer
    )

    # Print detection results
    print("Detection Results (Custom Method):")
    print(f"P-value: {p_value:.4f}")
    print(f"Result: {result}")

    print("Finished running custom method.")

def main():
    prompt = "This is a test prompt."
    model_name = "facebook/opt-1.3b"
    vocab_size = 50265
    n = 256
    m = 50
    k = 50  # Number of hash functions
    b = 20
    seed = 42  # Random seed
    fraction = 0.5
    strength = 2.0
    wm_key = 42
    max_new_tokens = 50
    output_dir = "./output"

    # File paths for the EXP, KGW, and custom scripts
    exp_generate_file = "generation.py"  # Replace with actual path to your EXP generation script
    exp_detect_file = "detection.py"  # Replace with actual path to your EXP detection script
    kgw_generate_file = "Unigram-Watermark-main/run_generate.py"  # Replace with actual path to your KGW generation script
    kgw_detect_file = "Unigram-Watermark-main/run_detect.py"  # Replace with actual path to your KGW detection script
    custom_generate_file = "generate_simhash.py"  # Replace with actual path to your custom generation script
    custom_detect_file = "detection_simhash.py"  # Replace with actual path to your custom detection script

    # Run all three methods
    run_exp(prompt, model_name, vocab_size, n, m, k, seed)  # Pass all required arguments for EXP
    # run_exp(prompt, model_name, exp_generate_file, exp_detect_file, vocab_size, n, m, k, seeds)
    # run_kgw(prompt, model_name, kgw_generate_file, kgw_detect_file, fraction, strength, wm_key, max_new_tokens, output_dir)
    run_custom(prompt, model_name, custom_generate_file, custom_detect_file, seed, k, b, m, n, vocab_size)

if __name__ == "__main__":
    main()
