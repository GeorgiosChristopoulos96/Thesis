import pandas as pd
import json
import os
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle


# Function to read JSON and preprocess data
def preprocess_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = data['entries']
    processed_data = []
    for entry_id, entry in enumerate(entries, 1):
        for lex in entry[f"{entry_id}"]["lexicalisations"]:
            modified_tr = entry[f"{entry_id}"]["modifiedtripleset"][0]
            triple_set = " | ".join(
                [f"{modified_tr['subject']} | {modified_tr['property']} | {modified_tr['object']}"])
            triple_set = triple_set.replace("_", " ")  # Replace underscores with spaces
            lex_text = lex['lex'].replace("_", " ")  # Replace underscores with spaces
            processed_data.append(
                [entry_id, entry[f"{entry_id}"]['category'], entry[f"{entry_id}"]['shape'], entry[f"{entry_id}"]['shape_type'], triple_set,
                 lex_text])

    return pd.DataFrame(processed_data, columns=['id', 'category', 'shape', 'shape_type', 'triples', 'lexicalization'])


# Directories
orig = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_En/release_v3.0'
prep = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_En/release_v3.0/prep'
os.makedirs(prep, exist_ok=True)

# Process splits
splits = ['train', 'dev','test']
for split in splits:
    file_path = f'{orig}/en/{split}/{split}.json'
    df = preprocess_json(file_path)

    # Shuffle train and dev splits
    if split in ['train', 'dev']:
        df = shuffle(df)

    # Save to TSV
    out_path = f'{prep}/{split}.tsv'
    df.to_csv(out_path, sep='\t', index=False)


# Tokenization (example using NLTK)
def tokenize_files(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as src_f, open(tgt_file, 'r', encoding='utf-8') as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()

    tokenized_src = [" ".join(word_tokenize(line)) for line in src_lines]
    tokenized_tgt = [" ".join(word_tokenize(line)) for line in tgt_lines]

    return tokenized_src, tokenized_tgt

