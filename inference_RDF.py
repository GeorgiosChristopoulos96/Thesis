import json
import os
import re
import subprocess

from tqdm import tqdm
import evaluate
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader

from Utils import format_only_labels, postprocess_text, load_eval_metrics, is_number, load_test_data_for_language

if not torch.backends.mps.is_available():
    print("MPS not available")
    device = torch.device("cpu")  # Fall back to CPU if MPS is not available
else:
    device = torch.device("mps")
# Assuming FLAGS is a dictionary containing configuration like batch size and model path
FLAGS = {
    'saved_model_path': '/Users/georgioschristopoulos/PycharmProjects/Thesis/saved_models/rdf_to_text_model_final',
    'batch_size': 4, # Adjust according to your setup
    'parent_script_path': '/Users/georgioschristopoulos/PycharmProjects/Thesis/PARENT_metric/table_text_eval/table_text_eval.py',  # Add the path to the PARENT script
    'output_dir': '/Users/georgioschristopoulos/PycharmProjects/Thesis',
}


def save_decoded_data(decoded_preds, decoded_labels, preds_file='decoded_preds.json', labels_file='decoded_labels.json'):
    with open(preds_file, 'w') as pf:
        json.dump(decoded_preds, pf)


def preprocess_function(examples):
    inputs = examples['input_text']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # Tokenize the targets as before
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length",
                           return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# def load_dataset_from_tsv_with_multiple_refs(tsv_path):
#     df = pd.read_csv(tsv_path, delimiter='\t', keep_default_na=False)
#     df['input_text'] = df['triples']
#     # Split the 'lexicalization' field into multiple references
#     df['target_texts'] = df['lexicalization'].apply(lambda x: x.split('.",'))
#     # Explode the DataFrame so that each RDF set gets multiple rows, one for each reference
#     df_exploded = df.explode('target_texts').rename(columns={'target_texts': 'target_text'})
#     df_exploded['target_text'] = df_exploded['target_text'].str.strip()
#     return Dataset.from_pandas(df_exploded[['input_text', 'target_text']])




def get_saved_model():
    """
    Retrieves the best model and tokenizer that was saved after fine-tuning.
    """
    saved_model = AutoModelForSeq2SeqLM.from_pretrained(FLAGS['saved_model_path'], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS['saved_model_path'], local_files_only=True, add_prefix_space=True)

    return saved_model.to("cuda" if torch.cuda.is_available() else "mps"), tokenizer

def generate_predictions(saved_model, test_set,lang):
    ""
    encoded_inputs = test_set.remove_columns("labels")
    # set-up a dataloader to load in the tokenized test dataset
    dataloader = torch.utils.data.DataLoader(encoded_inputs,  batch_size=FLAGS['batch_size'])
    language_prompts = {
        'br': 'RDF-to-br:',
        'cy': 'RDF-to-cy:',
        'ga': 'RDF-to-ga:',
        'mt': 'RDF-to-mt:'
    }
    language_prompt = language_prompts.get(lang, '')
    all_predictions = []
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Generating predictions in {lang}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_input = [language_prompt + text for text in
                       tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)]
        batch_encoded = tokenizer(batch_input, return_tensors="pt", padding=True).to(device)
        predictions = saved_model.generate(**batch_encoded, do_sample = True, max_new_tokens=100, top_p=0.7, repetition_penalty = 1.3)
        all_predictions.extend(predictions)
    decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True).replace('"', '') for pred in all_predictions]

    print('LOGGING: generate_predictions DONE \n')

    with open(f'decoded_predictions_{lang}.txt',  'w', encoding='utf-8') as file:
        for prediction in decoded_predictions:
            file.write(prediction + '\n')

    print('LOGGING: writing to file decode_predictions DONE \n')
    return decoded_predictions



# def format_only_labels(labels):
#     formatted_labels = []
#     for label in labels:
#         # Check if the label is already in the desired format: a list with a single string element
#         if isinstance(label, list) and len(label) == 1 and isinstance(label[0], str):
#             formatted_label = label
#         else:
#             # This block will handle cases where the label is not in the desired format,
#             # including when it's a list with multiple elements or contains brackets
#             if isinstance(label, list):
#                 # Concatenate elements, remove brackets, and strip whitespace
#                 formatted_label = [' '.join(label).replace('[', '').replace(']', '').strip()]
#             else:
#                 # For single string elements, just ensure no brackets and strip
#                 formatted_label = [label.replace('[', '').replace(']', '').strip()]
#         formatted_labels.append(formatted_label)
#     return formatted_labels
# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = format_only_labels(labels)
#     return preds, labels
# def load_eval_metrics():
#     """
#     Loads in all metrics that will be used later on during evaluation. This is seperated to not load in the metrics a dozen of times during training.
#     """
#     bleu = evaluate.load("sacrebleu")
#     rouge = evaluate.load('rouge')
#     meteor = evaluate.load('meteor')
#     ter = evaluate.load('ter')
#     #perplexity = evaluate.load("perplexity", module_type="metric")
#     bertscore = evaluate.load("bertscore")
#
#     print('LOGGING: load_eval_metrics DONE \n')
#
#     return bleu, rouge, meteor, ter,bertscore #perplexity, bertscore

def evaluate_texts(decoded_preds, decoded_labels):
    """
    Calculates metrics given a list of decoded predictions and decoded labels
    """

    #post_process for BLEU
    blue_preds, blue_labels = postprocess_text(decoded_preds,  decoded_labels)

    # setup metrics for use
    bleu, rouge, meteor, ter, bertscore = load_eval_metrics()
    log_filename = '/Users/georgioschristopoulos/PycharmProjects/Thesis/logs/metrics_log.txt'
    #Calculate the metrics
    print(f'\n LOGGING: Calculating Blue')
    bleu_output = bleu.compute(predictions=blue_preds, references=blue_labels)
    print(f'\n LOGGING: Calculating Rouge')
    rouge_output = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    print(f'\n LOGGING: Calculating Meteor')
    meteor_output = meteor.compute(predictions= decoded_preds, references=decoded_labels)
    print(f'\n LOGGING: Calculating TER')
    ter_output = ter.compute(predictions= decoded_preds, references=decoded_labels)
    print(f'\n LOGGING: Calculating Bertscore')
    bertscore_out= bertscore.compute(predictions= decoded_preds, references=decoded_labels, lang=lang)
    P = bertscore_out['precision']
    R = bertscore_out['recall']
    F1 = bertscore_out['f1']
    # Convert to list if they are not and calculate the mean
    P = float(sum(list(P)) / len(P))
    R = float(sum(list(R)) / len(R))
    F1 = float(sum(list(F1)) / len(F1))


    with open(log_filename, 'a') as log_file:

        log_file.write(f"\n LOGGING: Calculating Bleu\n")
        log_file.write(f"BLEU: {bleu_output}\n")
        print(f'\n LOGGING: Calculating Rouge')
        log_file.write(f"Rouge: {rouge}\n")
        print(f'\n LOGGING: Calculating Meteor')
        log_file.write(f"Rouge: {meteor_output}\n")
        print(f'\n LOGGING: Calculating TER')
        log_file.write(f"Rouge: {ter_output}\n")
        log_file.write(f"\n LOGGING: Calculating Bertscore\n")
        log_file.write(f"Bertscore Precision Mean: {P}\n")
        log_file.write(f"Bertscore Recall Mean: {R}\n")
        log_file.write(f"Bertscore F1 Mean: {F1}\n")

    print(f'\n LOGGING: Done calculations')
    return bleu_output, rouge_output, meteor_output, ter_output, F1, P, R
# Note: You'll need to adapt `evaluate_test_set` based on your specific evaluation functions and logging mechanism.
def evaluate_test_set(decoded_preds, true_articles_test,lang):
    """
    Transforms test set, retrieves predictions, and evaluates these predictions
    """
    bleu_output, rouge_output, meteor_output, ter, F1,P,R = evaluate_texts(decoded_preds, true_articles_test)

    # Huggingsface trainer requires a dict if multiple metrics are used
    evaluation_results = {"blue_output": bleu_output, "rouge_output": rouge_output, "meteor_results": meteor_output,"ter_results": ter,
                           "bertscore_F1": F1,"Bertscore_P": P,"Bertscore_R": R}



    return evaluation_results

# Assume 'tables' is your list from the screenshot
def prepare_inputs_parent(rdfs):
    """
    Cleans the RDF pairs and transforms them into the proper format so that the PARENT module can calculate with it.
    Input: RDF pairs of format "Attribute <s> Value <p> Relation <o>"
    Returns a list of lists containing tuples --> [ [(Attribute, Relation, Value), ...] ...]
    """
    attribute_relation_value_triples = []
    # .replace('<S>', '')\
    # .replace('<P>', '')\
    # .replace('<O>', '')\
    formatted_rdfs = []
    for rdf_string in rdfs:
        # Clean up the RDF string
        cleaned_rdf_string = rdf_string.replace('RDF-to-text: ', '')\
                                       .replace('[', '')\
                                       .replace(']', '')\
                                       .strip()

        triples = cleaned_rdf_string.split(', ')  # Assuming the triples are separated by commas
        formatted_triples = []
        for triple in triples:
            # Splitting the cleaned string into its constituent parts  # Assuming each triple is separated by '|'
            components = triple.split(' <O> ')
            if len(components) == 2:
                object_component = components[1]
                subject_predicate = components[0].split(' <P> ')
                if len(subject_predicate) == 2:
                    subject_component = subject_predicate[0].strip('<S> ')
                    predicate_component = subject_predicate[1]
                    formatted_triples.append(f"{subject_component}|||{predicate_component}|||{object_component}")
            #formated_rds.append([[predicate_component],[subject_component, object_component]])
        formatted_rdfs.append("\t".join(formatted_triples))
    return formatted_rdfs
def write_rdfs_to_file(rdfs, file_path):
    """
    Writes the formatted RDF triples to a text file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for rdf in rdfs:
            f.write(rdf + "\n")
def evaluate_individual_rdf(decoded_predictions, true_articles_test, rdf_triples,lang_code):
    gen_file = os.path.join(FLAGS['output_dir'], f'gen_{lang_code}.txt')
    ref_file = os.path.join(FLAGS['output_dir'], f'ref_{lang_code}.txt')
    table_file = os.path.join(FLAGS['output_dir'], f'table_{lang_code}.txt')

    with open(gen_file, 'w') as gf, open(ref_file, 'w') as rf, open(table_file, 'w') as tf:
        for pred, ref, rdf in zip(decoded_predictions, true_articles_test, rdf_triples):
            gf.write(pred + '\n')
            rf.write(ref + '\n')
            tf.write(rdf + '\n')

    result = run_parent_evaluation(gen_file, ref_file, table_file)
    return result


def run_parent_evaluation(generation_file, reference_file, table_file):
    """
    Run the PARENT metric evaluation script for the given files.
    """
    parent_command = [
        "python", FLAGS['parent_script_path'],
        "--references", reference_file,
        "--generations", generation_file,
        "--tables", table_file
    ]
    #parent_command = f"python {FLAGS['parent_script_path']} --references {reference_file} --generations {generation_file} --tables {table_file}"
    result = subprocess.run(parent_command,capture_output=True, text=True)
    if result.returncode != 0:
        print("PARENT evaluation script failed")
        print(result.stderr)
        return {}

        # Parse the JSON from the stdout
    try:
        output = result.stderr.split("]")[-1]

        numbers = re.findall(r"\d+\.\d+", output)

        # Keep only the numerical values
        data_list = [x for x in numbers if is_number(x)]
        data_dict = {"PARENT_precision": data_list[0], "PARENT_recall": data_list[1],
                     "PARENT_f1-score": data_list[2]}  # f1-score is more common than F-score

        # Convert the dictionary to JSON format
        parent_results = json.dumps(data_dict)
    except json.JSONDecodeError:
        print("Failed to parse PARENT evaluation results as JSON.")
        parent_results = {}

    return parent_results

    #return result
def transform_datasets(dataset):
    test_ds = dataset
    # to use the actual articles for evaluation
    true_articles_test = test_ds['target_text']
    # The Parent Metric requires the original RDFs
    test_rdf_input = test_ds['input_text']
    test_ds = test_dataset.map(preprocess_function,batched=True, remove_columns=test_dataset.column_names)
    # transform the datasets into torch sensors, as the model will expect this format
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    print('LOGGING: transform_datasets DONE \n')

    return  test_ds, true_articles_test, test_rdf_input



# Load the test set
if __name__ == "__main__":
    model, tokenizer = get_saved_model()

    languages = ['br', 'cy', 'ga','mt',]
    for lang in languages:
        # Load language-specific test data
        test_dataset = load_test_data_for_language(lang)
        test_dataset = test_dataset.select(range(5))
        test_ds, true_articles_test, test_rdf_input = transform_datasets(test_dataset)

        # Generate predictions
        predictions = generate_predictions(model, test_ds, lang)

        # Evaluate predictions using standard metrics
        evaluation_results = evaluate_test_set(predictions, true_articles_test,lang)
        if(lang =='mt'):
            evaluation_results =  {
                    "blue_output": evaluation_results["blue_output"],
                    "chrf_output": evaluation_results.get("chrf_output"),  # Assuming this is the key for chrF++
                    "ter_results": evaluation_results["ter_results"]
                                            }
        with open(f'evaluation_results_{lang}.json', 'w') as file:
            json.dump(evaluation_results, file)

        # PARENT evaluation
        # Ensure you have a function like `evaluate_individual_rdf` for PARENT evaluation
        rdf_triples = prepare_inputs_parent(test_rdf_input)  # Assuming this function prepares RDF triples correctly
        parent_results = evaluate_individual_rdf(predictions, true_articles_test, rdf_triples, lang)

        write_rdfs_to_file(rdf_triples, f'/Users/georgioschristopoulos/PycharmProjects/Thesis/tables{lang}.txt')
        parent_results = json.loads(parent_results)
        #parent_results = {k: sum(v) / len(v) for k, v in parent_results.items() if v}
        combined_results = {**evaluation_results, **parent_results}

        # Now write the combined results to a language-specific file
        with open(f'combined_evaluation_results_{lang}.json', 'w') as file:
            json.dump(combined_results, file)






