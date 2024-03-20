import json
import os
import re

def decode_unicode_escapes(value):
    return re.sub(r'\\u([0-9a-fA-F]{4})', lambda match: chr(int(match.group(1), 16)), value)

def clean_json_data(data, seen_translations=None):
    if seen_translations is None:
        seen_translations = set()

    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'translation':
                new_translations = []
                for translation_pair in value:
                    if len(translation_pair) == 2:
                        source_text, target_text = translation_pair.values()

                        # Check for non-identical source and target
                        if source_text != target_text:
                            pair_tuple = (source_text, target_text)
                            seen_translations.add(pair_tuple)
                            new_translations.append(translation_pair)
                        else:
                            print(f"Identical source and target removed: {translation_pair}")

                data[key] = new_translations
            else:
                data[key] = clean_json_data(value, seen_translations)
    elif isinstance(data, list):
        data = [clean_json_data(item, seen_translations) for item in data]
    elif isinstance(data, str):
        data = decode_unicode_escapes(data)
    return data

def clean_files_in_directory(directory_path):
    for file_name in ['train.json', 'test.json', 'validation.json']:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            print(f"Cleaning {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            cleaned_data = clean_json_data(data)

            cleaned_file_path = os.path.join(directory_path, f"cleaned_{file_name}")
            with open(cleaned_file_path, 'w', encoding='utf-8') as file:
                json.dump(cleaned_data, file, indent=4, ensure_ascii=False)

# Example usage
root_directory = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/test'  # Adjust the path to your dataset
for language_pair in os.listdir(root_directory):
    language_pair_path = os.path.join(root_directory, language_pair)
    if os.path.isdir(language_pair_path):
        clean_files_in_directory(language_pair_path)
