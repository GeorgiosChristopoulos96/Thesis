import json
from uuid import uuid4

# Original JSON data
original_json = json.load(open('/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/Maltese/am-en/validation.json', 'r', encoding='utf-8'))

# Reversing the order within each translation dictionary
reversed_order_json = {
    "translation": [
        {"en": item["en"], "am": item["am"]} for item in original_json["translation"]
    ]
}
filename = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/Maltese/en-am/validation.json'
with open(filename, 'w', encoding='utf-8') as file:
    json.dump(reversed_order_json, file, ensure_ascii=False, indent=4)

