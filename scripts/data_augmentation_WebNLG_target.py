import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# Setting up the device for PyTorch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the dataset from the provided CSV file
def load_data(file_path):
    return pd.read_csv(file_path, delimiter='\t', header=0)  # Adjust delimiter and header if needed

# Save data to CSV
def save_data(data, file_path):
    data.to_csv(file_path, index=False)

# Initialize model and tokenizer for translation
def create_model_tokenizer(model_name, src_lang_code, tgt_lang_code):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer

# Function to perform translation
def translate_text_batch(texts, model, tokenizer, target_lang_token, batch_size=32):
    translated_texts = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = inputs.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_token], max_length=128)
        batch_translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_texts.extend(batch_translated)
    return translated_texts

# Main translation process
def main_translation_process(data, model_name, language_pairs, sample_size_per_language, batch_size):
    for src, tgt in language_pairs:
        model, tokenizer = create_model_tokenizer(model_name, src, tgt)
        subset = data.sample(n=sample_size_per_language)
        texts = subset['lexicalization'].tolist()
        translated_texts = translate_text_batch(texts, model, tokenizer, tgt, batch_size)
        # Create new rows for each translated text and add to the dataframe
        for original_index, translated_text in zip(subset.index, translated_texts):
            new_row = data.loc[original_index].copy()
            new_row['lexicalization'] = translated_text
            if tgt == 'cym_Latn':
                new_row['prefix'] = 'Welsh'
            elif tgt == 'mlt_Latn':
                new_row['prefix'] = 'Maltese'
            elif tgt == 'gle_Latn':
                new_row['prefix'] = 'Irish'
            else:
                new_row['prefix'] = tgt
            data = data.append(new_row, ignore_index=True)
    return data

# Define paths and settings
PATH = '/Users/georgioschristopoulos/PycharmProjects/Thesis'
data_file_path = f'{PATH}/Datasets/WebNLG_MIXED(RU&EN)/combined_train.tsv'  # Path to your dataset
output_file_path = f'{PATH}/Datasets/WebNLG_MIXED(RU&EN)/synthetic_train.tsv'  # Path for the extended dataset

data = load_data(data_file_path)
model_name = "facebook/nllb-200-distilled-600M"
language_pairs = [('en', 'cym_Latn'), ('en', 'mlt_Latn'), ('en', 'gle_Latn')]
sample_size_per_language = 1500
batch_size = 32  # Adjust based on your environment's capability

data = main_translation_process(data, model_name, language_pairs, sample_size_per_language, batch_size)
filt = data[~data['prefix'].isin(['English', 'Russian'])]
save_data(filt, output_file_path)

print("Translation completed and data saved.")
