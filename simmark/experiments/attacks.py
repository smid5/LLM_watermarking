import torch
from transformers import MarianMTModel, MarianTokenizer, BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer
import os

from sentence_transformers import SentenceTransformer, util
import Levenshtein
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize

from transformers import logging
logging.set_verbosity_error()

distortion_model = SentenceTransformer('all-MiniLM-L6-v2')
    
def modify_text(tokenizer, vocab_size, text, num_modify):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()
    modified_ids = []
    torch.seed()

    num_modify = min(num_modify, len(ids))  

    for _ in range(num_modify):
        idx = torch.randint(0, len(ids), (1,))
        while idx[0] in modified_ids:
            idx = torch.randint(0, len(ids), (1,))  # Keep searching for a new index

        random_token = torch.randint(0, vocab_size, (1,))
        ids[idx] = random_token
        modified_ids.append(idx[0])

    text = tokenizer.decode(ids, skip_special_tokens=True)

    return text

def measure_distortion(original, modified):
    emb = distortion_model.encode([original, modified], convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(emb[0], emb[1]).item()
    edit_ratio = Levenshtein.distance(original, modified) / max(len(original), len(modified), 1)
    return round(cosine_sim, 4), round(edit_ratio, 4)

def translate_text(text, translate_whole = True, num_modify = None, language = "french"):
    """
    translate_whole=True is preferred
    """

    if language == "french":
        en_ne_model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
        ne_en_model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"
    elif language == "russian":
        #not working very well, occasionally hallucinates
        en_ne_model_name = "Helsinki-NLP/opus-mt-en-ru"
        ne_en_model_name = "Helsinki-NLP/opus-mt-ru-en"
    elif language == "german":
        en_ne_model_name = "Helsinki-NLP/opus-mt-en-de"
        ne_en_model_name = "Helsinki-NLP/opus-mt-de-en"

    en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
    en_ne_model = MarianMTModel.from_pretrained(en_ne_model_name)
    ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
    ne_en_model = MarianMTModel.from_pretrained(ne_en_model_name)

    if translate_whole:
        # Translate from English → Target Language
        tokens = en_ne_tokenizer(text, return_tensors="pt", padding=True)
        translated_tokens = en_ne_model.generate(**tokens)
        translated_text = en_ne_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # Translate back from Target Language → English
        tokens = ne_en_tokenizer(translated_text, return_tensors="pt", padding=True)
        roundtrip_token = ne_en_model.generate(**tokens)
        roundtrip_text = ne_en_tokenizer.decode(roundtrip_token[0], skip_special_tokens=True)

        cos_sim, edit_ratio = measure_distortion(text, roundtrip_text)
        log_path = "logs/original_vs_translated.txt"
        os.makedirs("logs", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("[TranslateText]\n")
            f.write(f"Original:   {text.strip()}\n")
            f.write(f"Translated: {roundtrip_text.strip()}\n")
            f.write(f"Semantic similarity: {cos_sim}, Edit ratio: {edit_ratio}\n")
            f.write("=" * 60 + "\n\n")
        return roundtrip_text

    else: #word-by-word translation
        # ids = tokenizer.encode(text, return_tensors="pt").squeeze()
        # word_list = [tokenizer.decode(id) for id in ids]
        # print(f"original: {word_list}")

        lines = text.splitlines(keepends=True)
        tokenized_lines = [word_tokenize(line) for line in lines]
        # Flatten tokens and keep index mapping
        flat_tokens = []
        position_map = []  # Stores (line_idx, token_idx) to reconstruct later
        for line_idx, line_tokens in enumerate(tokenized_lines):
            for token_idx, token in enumerate(line_tokens):
                flat_tokens.append(token)
                position_map.append((line_idx, token_idx))

        words_only_indices = [i for i, token in enumerate(flat_tokens) if token.isalpha()]
        if num_modify == None:
            num_modify = len(words_only_indices)
        num_modify = min(num_modify, len(words_only_indices))  

        random.shuffle(words_only_indices)
        selected_indices = words_only_indices[:num_modify]

        for idx in selected_indices:
            original_word = flat_tokens[idx]

            # Skip punctuation or short tokens
            if not original_word.isalpha():
                continue
            
            # Translate from English → Target Language
            token = en_ne_tokenizer(original_word, return_tensors="pt", padding=True)
            translated_token = en_ne_model.generate(**token, max_new_tokens=5)
            translated_text = en_ne_tokenizer.decode(translated_token[0], skip_special_tokens=True)

            # Translate back from Target Language → English
            token = ne_en_tokenizer(translated_text, return_tensors="pt", padding=True)
            roundtrip_token = ne_en_model.generate(**token, max_new_tokens=5)
            roundtrip_text = ne_en_tokenizer.decode(roundtrip_token[0], skip_special_tokens=True)

            flat_tokens[idx] = roundtrip_text

        # Reconstruct tokenized_lines from modified flat_tokens
        for i, token in enumerate(flat_tokens):
            line_idx, token_idx = position_map[i]
            tokenized_lines[line_idx][token_idx] = token

        detokenizer = TreebankWordDetokenizer()
        detokenized_lines = []
        for line_tokens, original_line in zip(tokenized_lines, lines):
            detok = detokenizer.detokenize(line_tokens)
            if original_line.endswith('\n'):
                detok += '\n'
            detokenized_lines.append(detok)

        # Reconstruct final sentence
        sentence = ''.join(detokenized_lines)
        # ids = tokenizer.encode(sentence, return_tensors="pt").squeeze()
        # word_list = [tokenizer.decode(id) for id in ids]
        # print(f"detokenized: {word_list}")
        # sentence = re.sub(r'\s+([.,!?;:])', r'\1', sentence)  
        # ids = tokenizer.encode(sentence, return_tensors="pt").squeeze()
        # word_list = [tokenizer.decode(id) for id in ids]
        # print(f"final: {word_list}")

        return sentence
    
def mask_modify_text(og_tokenizer, text, num_modify):
    # print(text)
    # ids = og_tokenizer.encode(text, return_tensors="pt").squeeze()
    # word_list = [og_tokenizer.decode(id) for id in ids]
    # print(f"start: {word_list}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForMaskedLM.from_pretrained('bert-base-cased')
    model.eval()

    lines = text.splitlines(keepends=True)
    tokenized_lines = [tokenizer.tokenize(line) for line in lines]
    # Flatten tokens and keep index mapping
    flat_tokens = []
    position_map = []  # Stores (line_idx, token_idx) to reconstruct later
    for line_idx, line_tokens in enumerate(tokenized_lines):
        for token_idx, token in enumerate(line_tokens):
            flat_tokens.append(token)
            position_map.append((line_idx, token_idx))

    words_only_indices = [i for i, token in enumerate(flat_tokens) if token not in tokenizer.all_special_tokens and token.isalpha()]

    num_modify = min(num_modify, len(words_only_indices))
    modified_indices = random.sample(words_only_indices, num_modify)

    for idx in modified_indices:
        original_token = flat_tokens[idx]
        # print(tokenizer.decode(tokenizer.convert_tokens_to_ids(original_token)))
        flat_tokens[idx] = '[MASK]'

        # Encode masked tokens
        input_ids = tokenizer.encode(flat_tokens, return_tensors='pt')
        mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits

        predicted_token_id = predictions[0, mask_idx].argmax().item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]

        flat_tokens[idx] = predicted_token

    for i, token in enumerate(flat_tokens):
        line_idx, token_idx = position_map[i]
        tokenized_lines[line_idx][token_idx] = token

    detokenized_lines = []
    for line_tokens, original_line in zip(tokenized_lines, lines):
        detok = tokenizer.convert_tokens_to_string(line_tokens)
        # Fix spacing around apostrophes (e.g., it ' s → it's)
        detok = re.sub(r"\s*'\s*", "'", detok)
        if original_line.endswith('\n'):
            detok += '\n'
        detokenized_lines.append(detok)
    sentence = ''.join(detokenized_lines)

    # ids = og_tokenizer.encode(sentence, return_tensors="pt").squeeze()
    # word_list = [og_tokenizer.decode(id) for id in ids]
    # print(f"final: {word_list}")
    
    return sentence
    
def delete_text(tokenizer, text, num_delete):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()
    torch.seed()
    
    num_delete = min(num_delete, len(ids))  
    delete_indices = torch.randperm(len(ids))[:num_delete]  # Select random indices to delete
    
    # Remove selected indices
    ids = torch.tensor([ids[i].item() for i in range(len(ids)) if i not in delete_indices], dtype=torch.long)
    
    # Ensure correct shape for decoding
    text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    
    return text

def insert_text(tokenizer, vocab_size, text, num_insert):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()
    torch.seed()
    
    for _ in range(num_insert):
        idx = torch.randint(0, len(ids) + 1, (1,))  # Choose a random position
        random_token = torch.randint(0, vocab_size, (1,))  # Generate a random token
        ids = torch.cat((ids[:idx], random_token, ids[idx:]))  # Insert it
    
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text

def duplicate_text(text, num_insert):
    words = text.split()
    num_insert = min(num_insert, len(words))  
    indices_to_duplicate = sorted(random.sample(range(len(words)), num_insert), reverse=True)

    for idx in indices_to_duplicate:
        words.insert(idx, words[idx])

    return ' '.join(words)