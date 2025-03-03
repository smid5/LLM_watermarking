import torch
from transformers import MarianMTModel, MarianTokenizer
    
def modify_text(tokenizer, vocab_size, text, num_modify):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()
    modified_ids = []

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

def translation_text(tokenizer, vocab_size, text, translate_whole = True, num_modify = None, language = "french"):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()

    if language == "french":
        en_ne_model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
        en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
        en_ne_model = MarianMTModel.from_pretrained(en_ne_model_name)

        ne_en_model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"
        ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
        ne_en_model = MarianMTModel.from_pretrained(ne_en_model_name)

        target_lang_id = "fr"
    elif language == "russian":
        en_ne_model_name = "Helsinki-NLP/opus-mt-en-ru"
        en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
        en_ne_model = MarianMTModel.from_pretrained(en_ne_model_name)

        ne_en_model_name = "Helsinki-NLP/opus-mt-ru-en"
        ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
        ne_en_model = MarianMTModel.from_pretrained(ne_en_model_name)

        target_lang_id = "ru"

    if translate_whole:
        # Translate from English → Target Language
        tokens = en_ne_tokenizer(text, return_tensors="pt", padding=True)
        translated_tokens = en_ne_model.generate(**tokens, forced_bos_token_id = target_lang_id)
        translated_text = en_ne_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # Translate back from Target Language → English
        tokens = ne_en_tokenizer(translated_text, return_tensors="pt", padding=True)
        roundtrip_token = ne_en_model.generate(**tokens, forced_bos_token_id = "en")
        roundtrip_text = ne_en_tokenizer.decode(roundtrip_token[0], skip_special_tokens=True)

        return roundtrip_text

    else: #word-by-word translation
        modified_ids = []

        if num_modify == None:
            num_modify = len(ids)
        num_modify = min(num_modify, len(ids))  

        for _ in range(num_modify):
            idx = torch.randint(0, len(ids), (1,))
            while idx[0] in modified_ids:
                idx = torch.randint(0, len(ids), (1,))  # Keep searching for a new index

            original_word = tokenizer.decode([ids[idx]], skip_special_tokens=True)

            # Translate from English → Target Language
            token = en_ne_tokenizer(original_word, return_tensors="pt", padding=True)
            translated_token = en_ne_model.generate(**token, max_new_tokens=1, forced_bos_token_id = target_lang_id)
            translated_text = en_ne_tokenizer.decode(translated_token[0], skip_special_tokens=True)

            # Translate back from Target Language → English
            token = ne_en_tokenizer(translated_text, return_tensors="pt", padding=True)
            roundtrip_token = ne_en_model.generate(**token, max_new_tokens=1, forced_bos_token_id = "en")
            roundtrip_text = ne_en_tokenizer.decode(roundtrip_token[0], skip_special_tokens=True)

            # Replace token with round-trip tokenized version
            new_token_id = tokenizer.encode(roundtrip_text, return_tensors="pt").squeeze()
            ids[idx] = new_token_id  # Ensure it's still a token ID tensor

            modified_ids.append(idx) 

        text = tokenizer.decode(ids, skip_special_tokens=True)
        return text
def delete_text(tokenizer, text, num_delete):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()
    
    num_delete = min(num_delete, len(ids))  
    delete_indices = torch.randperm(len(ids))[:num_delete]  # Select random indices to delete
    
    # Remove selected indices
    ids = torch.tensor([ids[i].item() for i in range(len(ids)) if i not in delete_indices], dtype=torch.long)
    
    # Ensure correct shape for decoding
    text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    
    return text

def insert_text(tokenizer, vocab_size, text, num_insert):
    ids = tokenizer.encode(text, return_tensors="pt").squeeze()
    
    for _ in range(num_insert):
        idx = torch.randint(0, len(ids) + 1, (1,))  # Choose a random position
        random_token = torch.randint(0, vocab_size, (1,))  # Generate a random token
        ids = torch.cat((ids[:idx], random_token, ids[idx:]))  # Insert it
    
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text

