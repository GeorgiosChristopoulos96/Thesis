import xml.etree.ElementTree as ET
from translate import Translator
def translate_text(text, src_lang, dest_lang):
    translator = Translator(to_lang = dest_lang, from_lang = src_lang)
    try:
        return translator.translate(text)
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # Return the original text in case of an error

def translate_xml(file_path, output_path, src_lang, dest_lang):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for lex in root.findall(".//lex"):
        text_element = lex.find('text')
        if text_element is not None and text_element.text:
            translated_text = translate_text(text_element.text, src_lang, dest_lang)
            text_element.text = translated_text
        text_element = lex.find('template')
        if text_element is not None and text_element.text:
            translated_text = translate_text(text_element.text, src_lang, dest_lang)
            text_element.text = translated_text

    tree.write(output_path, encoding='utf-8', xml_declaration=True)




# Example usage

translated_text = translate_text('Hello, world!', 'en', 'ro')
print(translated_text)

translate_xml('Airport.xml', 'TranslatedAirport.xml', 'en', 'ro')
