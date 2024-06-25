
import os
import json
import pandas as pd


def preprocess_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entries = data['entries']
    processed_data = []

    for entry_id, entry in enumerate(entries, 1):
        # Initialize triple_sets for each entry
        triple_sets = []

        for modified_tr in entry[f"{entry_id}"]["modifiedtripleset"]:
            # Construct each triple string
            triple = f"<S> {modified_tr['subject']} <P> {modified_tr['property']} <O> {modified_tr['object']}"
            # Replace underscores with spaces within the loop to ensure it's reset correctly
            triple = triple.replace("_", " ")
            triple_sets.append(triple)
        # Enclose the joined triples string in brackets
        triple_sets_str = "[" + ", ".join(triple_sets) + "]"

        # Process lexicalizations
        lex_texts = [lex['lex'].replace('_', ' ') for lex in entry[f"{entry_id}"]["lexicalisations"]]
        lex_texts_str = '", "'.join(lex_texts)
        # Encapsulate the entire lexicalization string in quotes
        lex_texts_str = f'"{lex_texts_str}"'

        processed_data.append(
            [entry_id, entry[f"{entry_id}"]['category'], entry[f"{entry_id}"]['shape'],
             entry[f"{entry_id}"]['shape_type'], triple_sets_str,
             lex_texts_str])

    return pd.DataFrame(processed_data, columns=['id', 'category', 'shape', 'shape_type', 'triples', 'lexicalization'])


# Directories
PATH = '/Users/georgioschristopoulos/PycharmProjects/Thesis'
orig = f'{PATH}/Datasets/WebNLG_Ru/release_v3.0/ru'
prep = f'{PATH}/Datasets/WebNLG_Ru/release_v3.0/ALT_prep'
os.makedirs(prep, exist_ok=True)

# Process splits
splits = ['train', 'dev', 'test']
for split in splits:
    file_path = f'{orig}/{split}/{split}.json'
    df = preprocess_json(file_path)

    # Shuffle train and dev splits
    if split in ['train', 'dev']:
        from sklearn.utils import shuffle
        df = shuffle(df)

    # Save to TSV
    out_path = f'{prep}/{split}.tsv'
    df.to_csv(out_path, sep='\t', index=False)
