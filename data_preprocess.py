import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import random

# Define the base and output directories
base_dir = Path("WebNLG_En/release_v3.0/en")
output_base = Path("WebNLG_En_preprocessed/release_v3.0/en")
references_dir = output_base / "references"

# Ensure the output directories exist
output_base.mkdir(parents=True, exist_ok=True)
references_dir.mkdir(parents=True, exist_ok=True)
lex_data = defaultdict(list)
def preprocess_xml_file(input_xml, output_triples_file, output_lex_file):
    # Ensure the output directory exists
    output_triples_file.parent.mkdir(parents=True, exist_ok=True)
    output_lex_file.parent.mkdir(parents=True, exist_ok=True)

    # Parse the XML file
    tree = ET.parse(input_xml)
    root = tree.getroot()
    # Open the output files
    with open(output_triples_file, 'w') as f_triples, open(output_lex_file, 'w') as f_lex:
        # Process each entry in the XML
        for entry in root.findall('.//entry'):
            mtriples = entry.find('.//modifiedtripleset')
            lexics = entry.findall('.//lex')

            if mtriples is not None and lexics is not None:
                # Extract and shuffle triples
                triples_list = [mtriple.text for mtriple in mtriples.findall('.//mtriple')]
                random.shuffle(triples_list)

                # Process each shuffled triple
                for triple in triples_list:
                    subject, predicate, object = triple.split(' | ')
                    subject = subject.lower()
                    predicate = predicate.lower()
                    object = object.strip('"').lower()
                    if("_" in subject):
                        subject = subject.replace("_", " ")
                    if("_" in object):
                        object = object.replace("_", " ")
                    if("_" in predicate):
                        predicate = predicate.replace("_", " ")
                    # Concatenate with distinct delimiters and write to the output triples file
                    f_triples.write(f"<S>{subject} <P>{predicate} <O>{object}\n")

                # Process and write lexicalizations
                for i,lex in enumerate(lexics):
                    text = lex.text.strip().replace('\n', ' ').replace('\t', ' ')
                    f_lex.write(f"{text}\n")
                    lex_data[i].append(f"{text}\n")


# Iterate over the directories and preprocess each XML file
for triples_dir in base_dir.glob('train/*triples'):
    print(f"Processing directory: {triples_dir.name}")
    output_dir = output_base / triples_dir.relative_to(base_dir)

    # Check if the directory is empty or doesn't exist
    if not triples_dir.exists() or not any(triples_dir.iterdir()):
        print(f"Directory {triples_dir} is empty or does not exist. Skipping.")
        continue

    # Process each XML file
    for file in triples_dir.glob('*.xml'):
        output_triples_file = output_dir / f"{file.stem}.triple.txt"
        output_lex_file = output_dir / f"{file.stem}.lex.txt"
        print(f"Preprocessing {file} -> {output_triples_file} and {output_lex_file}")
        preprocess_xml_file(file, output_triples_file, output_lex_file)
for i in lex_data:
    with open(references_dir / f"reference{i}.txt", 'w') as f:
        f.writelines(lex_data[i])
print("All directories processed. Preprocessed files are in", output_base)
