import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer
import os

from sentence_transformers import SentenceTransformer, util
import Levenshtein

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

def translate_text(tokenizer, vocab_size, text, translate_whole = True, num_modify = None, language = "french"):
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
        log_path = "original_vs_translated.txt"
        os.makedirs("logs", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("[TranslateText]\n")
            f.write(f"Original:   {text.strip()}\n")
            f.write(f"Translated: {roundtrip_text.strip()}\n")
            f.write(f"Semantic similarity: {cos_sim}, Edit ratio: {edit_ratio}\n")
            f.write("=" * 60 + "\n\n")
        return roundtrip_text

    else: #word-by-word translation
        ids = tokenizer.encode(text, return_tensors="pt").squeeze()
        modified_ids = []

        if num_modify == None:
            num_modify = len(ids)
        num_modify = min(num_modify, len(ids))  

        for _ in range(num_modify):
            idx = torch.randint(0, len(ids), (1,)).item()
            while idx in modified_ids:
                idx = torch.randint(0, len(ids), (1,)).item()  # Keep searching for a new index

            original_word = tokenizer.decode([ids[idx]], skip_special_tokens=True)
            # Translate from English → Target Language
            token = en_ne_tokenizer(original_word, return_tensors="pt", padding=True)
            translated_token = en_ne_model.generate(**token, max_new_tokens=5)
            translated_text = en_ne_tokenizer.decode(translated_token[0], skip_special_tokens=False)

            # Translate back from Target Language → English
            token = ne_en_tokenizer(translated_text, return_tensors="pt", padding=True)
            roundtrip_token = ne_en_model.generate(**token, max_new_tokens=5)
            roundtrip_text = ne_en_tokenizer.decode(roundtrip_token[0], skip_special_tokens=False)

            # Replace token with round-trip tokenized version
            new_token_id = tokenizer.encode(roundtrip_text, return_tensors="pt").squeeze()
            #ids[idx] = new_token_id[-1]  # Ensure it's still a token ID tensor
            ids = torch.cat((ids[:idx],new_token_id, ids[idx+1:]))

            for i in range(len(new_token_id)):
                modified_ids.append(idx+i) 

        ids = ids
        text = tokenizer.decode(ids, skip_special_tokens=True)
        return text
    
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
