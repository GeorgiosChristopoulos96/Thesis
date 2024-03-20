import nltk
import random
import string
from nltk.corpus import wordnet, stopwords
from random import choice
import json
nltk.download('wordnet')
nltk.download('stopwords')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)


def synonym_replacement(sentence, num_replacements=1):
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    non_stop_words = [word for word in words if word.lower() not in stop_words and len(get_synonyms(word)) > 0]

    for _ in range(num_replacements):
        if non_stop_words:
            word_to_replace = choice(non_stop_words)
            synonyms = get_synonyms(word_to_replace)
            if synonyms:
                synonym = choice(synonyms)
                words = [synonym if word == word_to_replace else word for word in words]
                non_stop_words.remove(word_to_replace)

    return ' '.join(words)


def inject_noise(sentence, noise_level=0.1):
    noisy_sentence = list(sentence)
    num_chars_to_change = max(1, int(len(sentence) * noise_level))

    for _ in range(num_chars_to_change):
        index_to_change = random.randint(0, len(sentence) - 1)
        replacement_char = random.choice(string.ascii_letters)
        noisy_sentence[index_to_change] = replacement_char

    return ''.join(noisy_sentence)


def augment_and_noisify_dataset(file_path, lang_code, num_augmentations=1, noise_level=0.1):
    data = load_json(file_path)
    augmented_data = {'translation': []}

    for entry in data['translation']:
        en_sentence = entry['en']

        # First augment with synonyms
        augmented_sentences = [synonym_replacement(en_sentence) for _ in range(num_augmentations)]

        # Then create noisy versions of both original and augmented sentences
        all_sentences = [en_sentence] + augmented_sentences
        for sentence in all_sentences:
            augmented_data['translation'].append({
                'en': sentence,
                f'{lang_code}': entry[lang_code]
            })
            noisy_sentence = inject_noise(sentence, noise_level)
            augmented_data['translation'].append({
                'en': noisy_sentence,
                f'{lang_code}': entry[lang_code]
            })

    # Save the augmented data to a new JSON file
    augmented_file_path = file_path.replace('.json', '_augmented_noisy.json')
    save_json(augmented_data, augmented_file_path)

    print(f"Augmented and noisified dataset saved to {augmented_file_path}")
    return augmented_file_path


# Example usage
lang_code = 'ga'
file_path = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/test_augmented/en-ga/train.json'
augmented_file_path = augment_and_noisify_dataset(file_path, lang_code, num_augmentations=1, noise_level=0.05)



def deduplicate_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    unique_entries = []
    seen = set()
    for item in data['translation']:
        # Convert both English and Gaelic sentences to lowercase before combining
        combined = f"{item['en'].lower()} {item[lang_code].lower()}"  # Ensure case-insensitive comparison
        if combined not in seen:
            unique_entries.append(item)
            seen.add(combined)

    deduplicated_file_path = file_path.replace('.json', '_deduplicated.json')
    with open(deduplicated_file_path, 'w', encoding='utf-8') as f:
        json.dump({'translation': unique_entries}, f, ensure_ascii=False, indent=4)

    print(f"Deduplicated dataset saved to {deduplicated_file_path}")

# Replace the path with your actual file path
deduplicate_dataset(augmented_file_path)

