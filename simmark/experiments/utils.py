
from ..methods import logit_processors, detection_methods
from transformers import LogitsProcessorList
import torch

def load_llm_config(model_name):
    if model_name == "facebook/opt-1.3b":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "vocab_size": 50272
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def extract_watermark_config(generation_name, watermark_config):
    method = generation_name.split("_")[0]
    watermark_config['method'] = method
    if method == "simmark":
        k, b = 10, 16
        if '_' in generation_name:
            k = int(generation_name.split("_")[1])
            b = int(generation_name.split("_")[2])
        watermark_config['k'] = k
        watermark_config['b'] = b
        watermark_config['prior_tokens'] = 10
    elif method == "expmin":
        hash_len = 3
        if '_' in generation_name:
            hash_len = int(generation_name.split("_")[1])
        watermark_config['hash_len'] = hash_len 

    elif method == "redgreen": pass
    elif method == "nomark": pass
    else:
        raise ValueError(f"Unknown generation method: {generation_name}")
    return watermark_config

def generate(text_start, num_tokens, llm_config, generation_name, seed=42):
    # Extract generation configuration
    gen_config = {
        'vocab_size': llm_config['vocab_size'],
        'model' : llm_config['model'],
        'seed' : seed,
    }
    gen_config = extract_watermark_config(generation_name, gen_config)
    
    input_ids = llm_config['tokenizer'].encode(text_start, padding=True, truncation=True, return_tensors="pt")

    logit_processor = logit_processors[gen_config['method']](gen_config)

    torch.manual_seed(gen_config['seed'])
    print('Starting generation...')
    outputs = llm_config['model'].generate(
        input_ids,
        max_new_tokens=num_tokens,
        logits_processor=LogitsProcessorList([logit_processor]),
        pad_token_id=llm_config['tokenizer'].eos_token_id
    )
    print('Generation complete!')

    return llm_config['tokenizer'].decode(outputs[0], skip_special_tokens=True)


def detect(text, llm_config, detection_name, seed=42):
    # Extract detection configuration
    detect_config = {
        'vocab_size': llm_config['vocab_size'],
        'tokenizer' : llm_config['tokenizer'],
        'model' : llm_config['model'],
        'seed' : seed,
    }
    detect_config = extract_watermark_config(detection_name, detect_config)
    
    p_value = detection_methods[detect_config['method']](text, detect_config)

    return p_value

def read_data(filename):
    # if filename doesn't exist, create it
    try:
        with open(filename, 'r') as f:
            pass
    except FileNotFoundError:
        with open(filename, 'w') as f:
            pass

    prompts, generated_texts, p_values, seeds = [], [], [], []
    with open(filename, 'r') as f:
        for line in f:
            line = eval(line)
            prompts.append(line['prompt'])
            generated_texts.append(line['generated_text'])
            p_values.append(line['p_value'])
            seeds.append(line['seed'])
    return {
        'prompts': prompts,
        'generated_texts': generated_texts,
        'p_values': p_values,
        'seeds': seeds
    }

def test_watermark(prompts, num_tokens, llm_config, generation_name, detection_name, attack_method=None, seed=42, folder='data/'):
    p_values = []
    filename = folder + f'{generation_name}_{detection_name}.txt'
    cached_data = read_data(filename)
    for prompt in prompts:
        # Check if prompt and seed is already in cached data
        if prompt in cached_data['prompts']:
            idx = cached_data['prompts'].index(prompt)
            if seed == cached_data['seeds'][idx]:
                p_values.append(cached_data['p_values'][idx])
                continue

        generated_text = generate(prompt, num_tokens, llm_config, generation_name, seed=seed)
        # Remove prompt from generated text
        generated_text = generated_text[len(prompt):]
        if attack_method is not None:
            generated_text = attack_method(generated_text)
        p_value = detect(generated_text, llm_config, detection_name, seed=seed)
        p_values.append(p_value)

        # Save output to file
        output = {
            'prompt': prompt,
            'generated_text': generated_text,
            'p_value': p_value,
            'seed' : seed
        }
        with open(filename, 'a') as f:
            f.write(str(output) + '\n')

    return p_values
