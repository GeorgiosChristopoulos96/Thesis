import re
import torch
from tqdm import tqdm
import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.corpus import wordnet
if not torch.backends.mps.is_available():
    print("MPS not available")
    mps_device = torch.device("cpu")  # Fall back to CPU if MPS is not available
else:
    mps_device = torch.device("mps")
def is_english_word(word):
    return bool(wordnet.synsets(word))
def clean_dataset(file_path, lang_code):
    data = load_json(file_path)
    clean_data = []

    for item in data['translation']:
        words = re.split(r'[\s,]+', item[lang_code])  # Split by whitespace and commas
        if item["en"] !=  item[lang_code] and not any(is_english_word(word) for word in words):
            clean_data.append(item)
    # Save the cleaned data back to the file or a new file
    cleaned_file_path = file_path.replace('.json', '_cleaned.json')
    save_json({'translation': clean_data}, cleaned_file_path)
    print(f"Cleaned dataset saved to {cleaned_file_path}")
    return clean_data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def batch_translate(texts, model, tokenizer, target_lang_token):
    if target_lang_token == "am":
        target_lang_token = "amh_Ethi"
    elif target_lang_token == "he":
        target_lang_token = "heb_Hebr"
    elif target_lang_token == "gd":
        target_lang_token = "gla_Latn"
    elif target_lang_token == "ga":
        target_lang_token = "gle_Latn"
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs.to(mps_device)
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=target_lang_token, max_length=128
        )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

def create_tokenizer_for_pair(src_lang_code, tgt_lang_code):
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src_lang_code, tgt_lang=tgt_lang_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(mps_device)
    return model, tokenizer

# Adjust the batch size as per your system's capabilities
batch_size = 50

# Model and tokenizer initialization
model_name = "facebook/nllb-200-distilled-600M"
models_tokenizers = {
    #'am': create_tokenizer_for_pair("eng_Latn", "amh_Ethi"),  # English to Amharic
    #'he': create_tokenizer_for_pair("eng_Latn", "heb_Hebr"),  # English to Hebrew
    'gd': create_tokenizer_for_pair("eng_Latn", "gla_Latn"),  # English to Scottish Gaelic
    # 'ga': create_tokenizer_for_pair("eng_Latn", "gle_Latn")   # English to Irish
    # Add other PARENT_metric pairs as needed
}

# Data directory and PARENT_metric pairs setup
data_dir = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/test'

# List of PARENT_metric pairs to iterate over
language_pairs = ['en-gd']#['en-am','en-he', 'en-gd', 'en-ga'] # Example PARENT_metric pairs

# Process each PARENT_metric pair
for lang_pair in language_pairs:
    combined_data = []
    target_lang_code = lang_pair.split('-')[1]
    lang_code =target_lang_code
    lang_dir = os.path.join(data_dir, lang_pair)
    if os.path.isdir(lang_dir):
        file_path = os.path.join(lang_dir, 'train.json')
        data = load_json(file_path)
        #data = clean_dataset(file_path, lang_code)
        model, tokenizer = models_tokenizers[target_lang_code]
        augmented_data = []
        for item in data['translation']:
            combined_entry = {
                'en': item['en'],
                lang_code: item[lang_code]  # Use the original text for the target PARENT_metric
            }
            combined_data.append(combined_entry)
        # Prepare batches of texts for translation
        texts = [item['en'] for item in data['translation']]
        if target_lang_code == "am":
            target_lang_code = "amh_Ethi"
        elif target_lang_code == "he":
            target_lang_code = "heb_Hebr"
        elif target_lang_code == "gd":
            target_lang_code = "gla_Latn"
        elif target_lang_code == "ga":
            target_lang_code = "gle_Latn"

        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            target_lang_token = tokenizer.lang_code_to_id[target_lang_code]  # Adjust the PARENT_metric code suffix
            translated_texts = batch_translate(batch_texts, model, tokenizer, target_lang_token)
            back_translated_texts = batch_translate(translated_texts, model, tokenizer, tokenizer.lang_code_to_id["eng_Latn"])
            for translated_text, back_translated_text in zip(translated_texts,back_translated_texts):
                    augmented_entry = {
                        'en': back_translated_text,
                        f'{lang_code}': translated_text
                        }
                    combined_data.append(augmented_entry)

        # Save the augmented data to a new JSON file in the same directory
        new_file_name = file_path.replace('.json', '_augmented.json')
        new_file_path = os.path.join(lang_dir, new_file_name)
        augmented_data_structure = {'translation': combined_data}
        save_json(augmented_data_structure, new_file_path)

        print("Back translation and data augmentation completed for selected PARENT_metric pairs.")

