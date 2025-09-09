
from ..methods import logit_processors, detection_methods
from simmark.experiments.attacks import modify_text, delete_text, insert_text, translate_text, mask_modify_text, duplicate_text
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

linestyles = ['dotted', 'solid', 'dashed', 'dashdot', (5,(10,3)), (0,(1,1)), (0,(5,10)),(0,(5,1)), (0,(3,10,1,10)), (0,(3,5,1,5)), (0,(3,1,1,1)), (0,(3,5,1,5,1,5)), (0,(3,10,1,10,1,10)), (0,(3,1,1,1,1,1))]

cbcolors = [
    '#4477AA',  # Medium blue
    '#44AA99',  # Teal
    '#CC6677',  # Soft red
    '#88CCEE',  # Cyan
    '#117733',  # Green
    '#332288',  # Dark blue
    '#661100',  # Brown
    '#882255',  # Dark red
    '#AA4466',  # Rose
    '#6699CC',  # Light blue
    '#AA4499',  # Purple
]

COLORS = {
    "SimMark": "#1f77b4",  # blue
    "ExpMin": "#ff7f0e",  # orange
    "SoftRedList": "#2ca02c",  # green
    "Unigram": "#d62728",  # red
    "SynthID": "#ff6ec7",  # pink
    "No Watermark": "#8c564b",  # brown
    "SimSoftRed": "#00FFFF", # cyan
    "SimSynthID": "#9467bd",  # purple
    "ExpMin (standard)": "#ff7f0e",  # orange
    "ExpMin (Simhash)": "#1f77b4",  # blue
    "SynthID (standard)": "#ff6ec7",  # pink
    "SynthID (Simhash)": "#d62728",  # red
}

METHODS = {
    "SimMark": "simmark",
    "ExpMin": "expmin",
    "SoftRedList": "softred",
    "Unigram": "unigram",
    "SynthID": "synthid",
    "No Watermark": "nomark",
    "SimSoftRed": "simsoftred",
    "SimSynthID": "simsynthid"
}

def load_prompts(filename):
    prompts = []
    with open(filename, 'r') as f: 
        for line in f:
            prompts.append(line.strip())
    return prompts

def load_llm_config(model_name):
    if model_name == "facebook/opt-125m":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "vocab_size": 50272
        }
    elif model_name == "meta-llama/Llama-3.2-3B":
        hf_token = "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"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     token=hf_token,
                                                     torch_dtype="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        return {
            "model": model,
            "tokenizer": tokenizer,
            "vocab_size": model.config.vocab_size
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def extract_watermark_config(generation_name, watermark_config):
    parts = generation_name.split("_")
    method = parts[0]
    watermark_config['method'] = method
    isSimHash = parts[1] == "simhash" if len(parts) > 1 else False
    watermark_config['isSimHash'] = isSimHash
    if method == "expmin":
        k = 4
        b = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if len(parts)>3:
            k = int(parts[2])
            b = int(parts[3])
        watermark_config['k'] = k
        watermark_config['b'] = b
        watermark_config['transformer_model'] = SentenceTransformer('all-MiniLM-L6-v2').to(device)
        watermark_config['prior_tokens'] = 8
    elif method == "simsoftred":
        n_gram = 2
        b = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if '_' in generation_name:
            n_gram = int(generation_name.split('_')[1])
            b = int(generation_name.split("_")[2])
        watermark_config['n_gram'] = n_gram
        watermark_config['b'] = b
        watermark_config['transformer_model'] = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    elif method == "synthid":
        k = 4
        b = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if len(parts)>3:
            k = int(parts[2])
            b = int(parts[3])
        watermark_config['depth'] = 30  # follow original paper
        watermark_config['prior_tokens'] = 4
        watermark_config['k'] = k
        watermark_config['b'] = b
        watermark_config['transformer_model'] = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    elif method == "softred": 
        n_gram = 2
        if '_' in generation_name:
            n_gram = int(generation_name.split('_')[1])
        watermark_config['n_gram'] = n_gram
    elif method == "unigram": pass
    elif method == "nomark": pass
    else:
        raise ValueError(f"Unknown generation method: {generation_name}")
    return watermark_config

def generate(text_start, num_tokens, llm_config, generation_name, seed=42, top_p=0.9):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Extract generation configuration
    gen_config = {
        'vocab_size': llm_config['vocab_size'],
        'model' : llm_config['model'],
        'seed' : seed,
        'tokenizer' : llm_config['tokenizer']
    }
    gen_config = extract_watermark_config(generation_name, gen_config)
    
    inputs = llm_config['tokenizer'](text_start, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    logit_processor = logit_processors[gen_config['method']](gen_config)

    torch.manual_seed(gen_config['seed'])
    print('Starting generation...')
    outputs = llm_config['model'].generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=num_tokens,
        do_sample=False,
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
    outputs = {}
    with open(filename, 'r') as f:
        for line in f:
            line = eval(line)
            for key in line:
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(line[key])
    return outputs


def extract_attack(llm_config, attack_name):
    attack_parts = attack_name.split('_')
    attack_type = attack_parts[0]
    num_changes = int(attack_parts[1]) if len(attack_parts) > 1 else 1  # Default to 1 change
    translate_whole = False if len(attack_parts) > 1 else True # Default to translating whole text, otherwise translate token-wise

    if attack_type == "modify":  # Substitution attack
        return lambda text: modify_text(
            llm_config['tokenizer'],
            llm_config['vocab_size'],
            text,
            num_modify=num_changes
        )
    elif attack_type == "delete":  # Deletion attack
        return lambda text: delete_text(
            llm_config['tokenizer'],
            text,
            num_delete=num_changes
        )
    elif attack_type == "insert":  # Insertion attack
        return lambda text: insert_text(
            llm_config['tokenizer'],
            llm_config['vocab_size'],
            text,
            num_insert=num_changes
        )
    elif attack_type == "translate": # Translation attack
        return lambda text: translate_text(
            text=text, 
            translate_whole=translate_whole,
            num_modify=num_changes
        )
    elif attack_type == "mask": # Mask modify attack
        return lambda text: mask_modify_text(
            og_tokenizer=llm_config['tokenizer'],
            text=text,
            num_modify=num_changes
        )
    elif attack_type == "duplicate": # Duplicate insertion attack
        return lambda text: duplicate_text(
            text=text,
            num_insert=num_changes
        )
    else:
        raise ValueError(f"Unknown attack method: {attack_name}")

def test_watermark(prompts, num_tokens, llm_config, generation_name, detection_name, attack_name="", seed=42, folder='data/'):
    p_values = []
    filename = folder + f'{generation_name}_{detection_name}_{attack_name}.txt'
    cached_data = read_data(filename)
    matches = ['prompt', 'seed', 'num_tokens']
    for prompt in prompts:
        # Check if prompt and seed is already in cached data
        try:
            # Find all indices where the prompt matches
            indices = [i for i, p in enumerate(cached_data['prompt']) if p == prompt]

            is_match = len(indices)>0
            # Check if any of those indices match seed and num_tokens
            for idx in indices:
                is_match = True
                for match in matches:
                    if cached_data[match][idx] != locals()[match]:
                        is_match = False
                if is_match:
                    p_values.append(cached_data['p_value'][idx])
                    break

            if is_match:
                continue  # Skip generating text if we found a cached match
        except (KeyError, ValueError):
            pass

        generated_text = generate(prompt, num_tokens, llm_config, generation_name, seed=seed)
        if attack_name != "":
            attack_method = extract_attack(llm_config, attack_name)
            generated_text = attack_method(generated_text)
        p_value = detect(generated_text, llm_config, detection_name, seed=seed)
        p_values.append(p_value)

        # Save output to file
        output = {
            'prompt': prompt,
            'generated_text': generated_text,
            'p_value': p_value,
            'seed' : seed,
            'num_tokens' : num_tokens,
        }
        with open(filename, 'a') as f:
            f.write(str(output) + '\n')

    return p_values

# def test_distortion(prompts, num_tokens, llm_config, generation_name, detection_name, attack_name="", seed=42, folder='data/'):
#     perplexity_list = []
#     filename = folder + f'{generation_name}_{detection_name}_{attack_name}.txt'
#     cached_data = read_data(filename)
#     matches = ['prompt', 'seed', 'num_tokens']
#     print(f"Calculating distortion for generator: {generation_name}, detector: {detection_name}, attack: {attack_name if attack_name else "None"}")
#     for prompt in prompts:
#         generated_text = ""
#         # Check if prompt and seed is already in cached data
#         try:
#             # Find all indices where the prompt matches
#             indices = [i for i, p in enumerate(cached_data['prompt']) if p == prompt]

#             is_match = len(indices)>0
#             # Check if any of those indices match seed and num_tokens
#             for idx in indices:
#                 is_match = True
#                 for match in matches:
#                     if cached_data[match][idx] != locals()[match]:
#                         is_match = False
#                 if is_match:
#                     generated_text = cached_data['generated_text'][idx]
#                     perplexity_list.append(sentence_perplexity(prompt, generated_text, llm_config))
#                     break

#             if is_match:
#                 continue  # Skip generating text if we found a cached match
#         except (KeyError, ValueError):
#             pass

#         generated_text = generate(prompt, num_tokens, llm_config, generation_name, seed=seed)
#         if attack_name != "":
#             attack_method = extract_attack(llm_config, attack_name)
#             generated_text = attack_method(generated_text)
#         p_value = detect(generated_text, llm_config, detection_name, seed=seed) # Calculate p_value to be stored for later use
#         # Save output to file
#         output = {
#             'prompt': prompt,
#             'generated_text': generated_text,
#             'p_value': p_value,
#             'seed' : seed,
#             'num_tokens' : num_tokens,
#         }
#         with open(filename, 'a') as f:
#             f.write(str(output) + '\n')
#         perplexity_list.append(sentence_perplexity(prompt, generated_text, llm_config))
    
#     return perplexity_list

def test_distortion(prompts, num_tokens, llm_config, generation_name, detection_name, attack_name="", seed=42, folder='data/'):
    perplexity_list = []

    # file where generated text is stored
    filename = folder + f'{generation_name}_{detection_name}_{attack_name}.txt'
    # file where perplexities are stored
    distortion_filename = folder + f'{generation_name}_{detection_name}_{attack_name}_distortion.txt'

    cached_data = read_data(filename)
    distortion_data = read_data(distortion_filename)
    matches = ['prompt', 'seed', 'num_tokens']
    print(f"Calculating distortion for generator: {generation_name}, detector: {detection_name}, attack: {attack_name if attack_name else 'None'}")
    for prompt in prompts:
        # First check if we already have the perplexity cached
        try:
            indices = [i for i, p in enumerate(distortion_data['prompt']) if p == prompt]

            is_match = len(indices) > 0
            for idx in indices:
                is_match = True
                for match in matches:
                    if distortion_data[match][idx] != locals()[match]:
                        is_match = False
                if is_match:
                    perplexity_list.append(distortion_data['perplexity'][idx])
                    break
            if is_match:
                continue  # Skip everything if perplexity already computed
        except (KeyError, ValueError):
            pass

        # Try reading from generation cache
        generated_text = ""
        try:
            indices = [i for i, p in enumerate(cached_data['prompt']) if p == prompt]
            is_match = len(indices) > 0
            for idx in indices:
                is_match = True
                for match in matches:
                    if cached_data[match][idx] != locals()[match]:
                        is_match = False
                if is_match:
                    generated_text = cached_data['generated_text'][idx]
                    break
        except (KeyError, ValueError):
            is_match = False

        # If not found, generate
        if not is_match:
            generated_text = generate(prompt, num_tokens, llm_config, generation_name, seed=seed)
            if attack_name != "":
                attack_method = extract_attack(llm_config, attack_name)
                generated_text = attack_method(generated_text)
            p_value = detect(generated_text, llm_config, detection_name, seed=seed)
            # Save generation + p_value
            output = {
                'prompt': prompt,
                'generated_text': generated_text,
                'p_value': p_value,
                'seed': seed,
                'num_tokens': num_tokens,
            }
            with open(filename, 'a') as f:
                f.write(str(output) + '\n')

        # Compute and save perplexity
        perp = sentence_perplexity(prompt, generated_text, llm_config)
        perplexity_list.append(perp)
        distortion_output = {
            'prompt': prompt,
            'perplexity': perp,
            'seed': seed,
            'num_tokens': num_tokens,
        }
        with open(distortion_filename, 'a') as f:
            f.write(str(distortion_output) + '\n')

    return perplexity_list
        
def sentence_perplexity(prompt, generated_text, llm_config):
    model = llm_config['model']
    tokenizer = llm_config['tokenizer']
    device = next(model.parameters()).device

    ids = llm_config['tokenizer'].encode(generated_text, return_tensors="pt").squeeze().to(device)
    input_ids = llm_config['tokenizer'].encode(prompt, return_tensors="pt").squeeze().to(device)
    token_probs = []

    while len(input_ids) != len(ids):
        with torch.no_grad():
            input_text = llm_config['tokenizer'].decode(input_ids, skip_special_tokens=True)
            input_tensor = llm_config['tokenizer'](input_text, return_tensors="pt")["input_ids"].to(device)
            original_logits = llm_config['model'](input_tensor).logits
            original_probs = torch.softmax(original_logits[0,-1,:], dim=-1)
            token_probs.append(torch.log(original_probs[ids[len(input_ids)].item()]))

            input_ids = torch.cat((input_ids, ids[len(input_ids)].unsqueeze(0)))

    perplexity = np.exp(-torch.stack(token_probs).cpu().mean().item())
    return perplexity