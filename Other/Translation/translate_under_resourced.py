import json

from translate import Translator
def translate_text(text, src_lang, dest_lang):
    translator = Translator(to_lang = dest_lang, from_lang = src_lang, de = "georgelos15@hotmail.co.uk")
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # Return the original text in case of an error


def translate_text_to_text(file_path, output_path, src_lang, dest_lang):
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Iterate through each item in the "translation" list
    for item in data["translation"]:
        # Extract the English text
        english_text = item[src_lang]
        # Translate from English to Amharic
        translated_text = translate_text(english_text, src_lang, dest_lang)
        # Update the Amharic text with the backtranslated text
        item[dest_lang] = translated_text

    # Save the updated data to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



# Example usage

translated_text = translate_text('Hello, world!', 'en', 'am')
print(translated_text)

translate_text_to_text('../Datasets/OPUS-100/Maltese/am-en/test.json', '../Datasets/OPUS-100/Maltese/am-en/BackTranslated.json', 'en', 'am')
