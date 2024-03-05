from datasets import load_dataset
import os
import json

# Define languages to filter for
Maltese_rel_languages = ["am-en", "ar-en", "en-it", "en-he", "en-fr"]
Celtic_langs = ["en-gd", "en-ga"]
opus_dir = "OPUS-100"
if not os.path.exists(opus_dir):
    os.makedirs(opus_dir)
else:
    print("OPUS-100 directory already exists")

# Download Maltese related languages
path = os.path.join(opus_dir, 'Celtic')
for lang in Celtic_langs:
    lang_dir = os.path.join(path, lang)
    if not os.path.exists(lang_dir):
        os.makedirs(lang_dir)
    opus_dataset = load_dataset("opus100", lang, data_dir=lang_dir)

    for split in opus_dataset.keys():
        data = opus_dataset[split]
        json_file_path = os.path.join(lang_dir, f"{split}.json")
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            # Convert to list of dictionaries (one per example)
            data_to_write = data.to_dict()
            # Save the list of dictionaries as JSON
            json.dump(data_to_write, json_file, ensure_ascii=False, indent=4)

print("Filtered languages downloaded and stored in OPUS-100 directory.")
