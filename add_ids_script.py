import json

input_file_path = './OPUS-100/Maltese/en-fr/test.json'
output_file_path = './OPUS-100/Maltese/en-fr/test_with_id.json'
'''
import uuid

# Generate a random UUID.
unique_id = uuid.uuid4()
print(unique_id)
'''

try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize the modified data structure
    modified_data = {"translation": []}
    i  = 0
    for item in data.get("translation", []):
        # Assuming each item is a dictionary that might have an 'id' and definitely has a 'translation' sub-dictionary
        if 'translation' in item:
            # Directly add the item if it follows the desired structure
            modified_data["translation"].append(item['translation'])
        else:
            # If the item is not structured correctly, adjust according to your needs
            # This part might need to be adjusted based on your actual data structure
            new_entry = {"en": item["en"], "fr": item["fr"]}
            new_entry = {'id': f'{i}', 'translation': new_entry}
            modified_data["translation"].append(new_entry)
            i += 1
    # Save the modified dataset
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(modified_data, file, ensure_ascii=False, indent=4)

    print("Dataset modified and saved successfully.")

except Exception as e:
    print(f"An error occurred: {e}")


