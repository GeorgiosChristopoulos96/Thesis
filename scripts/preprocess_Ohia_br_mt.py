import pandas as pd
import json
import os

# Function to read JSON and preprocess data
import json
import pandas as pd


def preprocess_json(file_path, target_lang='en'):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entries = data['entries']
    processed_data = []

    for entry_id, entry in enumerate(entries, 1):
        triple_sets = []

        for modified_tr in entry[f"{entry_id}"]["modifiedtripleset"]:
            triple = f"<S> {modified_tr['subject']} <P> {modified_tr['property']} <O> {modified_tr['object']}"
            triple = triple.replace("_", " ")
            triple_sets.append(triple)

        triple_sets_str = "[" + ", ".join(triple_sets) + "]"

        # Filter lexicalizations based on the target language
        lex_texts = [lex['lex'].replace('_', ' ') for lex in entry[f"{entry_id}"]["lexicalisations"] if
                     lex['lang'] == target_lang]

        if not lex_texts:  # If no lexicalizations in target_lang, skip this entry
            continue

        lex_texts_str = '", "'.join(lex_texts)
        lex_texts_str = f'"{lex_texts_str}"'

        processed_data.append([entry_id, entry[f"{entry_id}"]['category'], entry[f"{entry_id}"]['shape'],
                               entry[f"{entry_id}"]['shape_type'], triple_sets_str, lex_texts_str])

    return pd.DataFrame(processed_data, columns=['id', 'category', 'shape', 'shape_type', 'triples', 'lexicalization'])


# Directories
orig = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_br_mt_cy/2023-Challenge/data'
prep = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_br_mt_cy/2023-Challenge/data/ALT_prep'
os.makedirs(prep, exist_ok=True)

splits = ['train', 'dev']
languages = ['mt', 'ga', 'br', 'cy']  # Example languages

for split in splits:
    file_path = f'{orig}/{split}/{split}.json'
    for lang in languages:
        df = preprocess_json(file_path, target_lang=lang)
        if not df.empty:
            # Shuffle train and dev splits
            if split in ['train', 'dev']:
                df = df.sample(frac=1).reset_index(drop=True)

            # Save to TSV, naming file to indicate language
            out_path = f'{prep}/{split}_{lang}.tsv'
            df.to_csv(out_path, sep='\t', index=False)
