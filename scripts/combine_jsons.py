import os
import json
import uuid

base_path = "/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100"  # Replace with the actual path
exclude_dirs = ["am-en", "ar-en"]
aggregated_data = {
    "train": [],
    "test": [],
    "validation": []
}

# Function to create a unique ID
def generate_id():
    return str(uuid.uuid4())

# Traverse the directories
for root, dirs, files in os.walk(base_path):
    # Ignore specified directories
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file_name in files:
        if file_name in ["train.json", "test.json", "validation.json"]:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Assume data is a list of translation pairs like [{"en": "text", "am": "text"}]
                for record in data["translation"]:
                    # Add a unique ID to each record
                    lang1, lang2 = list(record.keys())
                    record_with_id = {
                        "id": generate_id(),
                        lang1: record[lang1],
                        lang2: record[lang2]
                    }
                    file_type = file_name.split('.')[0]
                    aggregated_data[file_type].append(record_with_id)

# Write out the aggregated files
for file_type, records in aggregated_data.items():
    final_data = {"translation": records}
    # Here, you could write final_data to a new JSON file or process it further
    output_path = os.path.join(base_path, f"aggregated_{file_type}.json")
    with open(output_path, 'w', encoding= 'utf-8') as outfile:
        json.dump(final_data, outfile, indent=4)
