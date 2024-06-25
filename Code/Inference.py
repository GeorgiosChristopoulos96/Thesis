import json
import os
import re
import subprocess
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from Utils import postprocess_text, load_eval_metrics, is_number, load_test_data_for_language, CUDA_CLEAN
device = "cuda" if torch.cuda.is_available() else "cpu"
PATH = "/"
SAVED_MODEL_PATH = f"{PATH}/RDF-to-text_2nd_exp"
PARENT_SCRIPT_PATH = f"{PATH}/PARENT_metric/table_text_eval/table_text_eval.py"
OUTPUT_DIR =PATH




def preprocess_function(examples):
    inputs = examples['input_text']
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length", return_tensors="pt")

    # Tokenize the targets as before
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length",
                           return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_saved_model():
    """
    Retrieves the best model and tokenizer that was saved after fine-tuning.
    """
    saved_model = AutoModelForSeq2SeqLM.from_pretrained(SAVED_MODEL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH, local_files_only=True, add_prefix_space=True)

    return saved_model.to("cuda" if torch.cuda.is_available() else "mps"), tokenizer

def generate_predictions(saved_model, test_set,lang):
    ""
    encoded_inputs = test_set.remove_columns("labels")
    # set-up a dataloader to load in the tokenized test dataset
    dataloader = torch.utils.data.DataLoader(encoded_inputs,  batch_size= 4)
    language_prompts = {
        'br': 'RDF-to-text in Breton:',
        'cy': 'RDF-to-text in Welsh:',
        'ga': 'RDF-to-text in Irish:',
        'mt': 'RDF-to-text in Maltese:'
    }
    language_prompt = language_prompts.get(lang, '')
    all_predictions = []
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Generating predictions in {lang}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_input = [language_prompt + text for text in
                       tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)]
        batch_encoded = tokenizer(batch_input, return_tensors="pt", padding=True).to(device)
        predictions = saved_model.generate(**batch_encoded, do_sample = True, max_new_tokens=200, num_beams=4,repetition_penalty = 3.5)
        all_predictions.extend(predictions)
    decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True).replace('"', '') for pred in all_predictions]

    print('LOGGING: generate_predictions DONE \n')

    with open(f'decoded_predictions_{lang}.txt',  'w', encoding='utf-8') as file:
        for prediction in decoded_predictions:
            file.write(prediction + '\n')

    print('LOGGING: writing to file decode_predictions DONE \n')
    return decoded_predictions


def evaluate_texts(decoded_preds, decoded_labels,language):
    """
    Calculates metrics given a list of decoded predictions and decoded labels
    """

    #post_process for BLEU
    formatted_preds, formatted_labels = postprocess_text(decoded_preds,  decoded_labels)
    adjusted_preds = [s.replace("'", '"') for s in formatted_preds]
    adjusted_labels = [[s.replace('"', '').replace("'", '') for s in sublist] for sublist in formatted_labels]


    # setup metrics for use
    bleu, rouge, meteor, ter, chrf, bertscore = load_eval_metrics()
     #Calculate the metrics
    print(f'\n LOGGING: Calculating Blue')
    bleu_output = bleu.compute(predictions=adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating Rouge')
    rouge_output = rouge.compute(predictions=adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating Meteor')
    meteor_output = meteor.compute(predictions= adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating TER')
    ter_output = ter.compute(predictions= adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating chrf')
    chrf_output = chrf.compute(predictions=adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating Bertscore')
    bertscore_out= bertscore.compute(predictions= adjusted_preds, references=adjusted_labels, lang=language)
    P = bertscore_out['precision']
    R = bertscore_out['recall']
    F1 = bertscore_out['f1']
    # Convert to list if they are not and calculate the mean
    P = float(sum(list(P)) / len(P))
    R = float(sum(list(R)) / len(R))
    F1 = float(sum(list(F1)) / len(F1))

    print(f'\n LOGGING: Done calculations')
    return bleu_output, rouge_output, meteor_output, ter_output, chrf_output, F1, P, R
# Note: You'll need to adapt `evaluate_test_set` based on your specific evaluation functions and logging mechanism.
def evaluate_test_set(decoded_preds, true_articles_test,lang):
    """
    Transforms test set, retrieves predictions, and evaluates these predictions
    """
    bleu_output, rouge_output, meteor_output, ter, chrf, F1,P,R = evaluate_texts(decoded_preds, true_articles_test,lang)
    metrics_evaluation = {"bleu": bleu_output["score"],
                          "rouge": rouge_output["rougeL"],
                          "meteor": meteor_output["meteor"],
                          "TER": ter,
                          "chrf": chrf["score"],
                          "bertscore_F1": F1, "bertscore_P": P,
                          "bertscore_R": R}
    filename = f"{PATH}/2nd_experiment_generated_files/evaluation_steps_metrics_inference.txt"

    with open(filename, "a") as file:
        file.write(f"Testing inference :\n")
        for key, value in metrics_evaluation.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")

    return metrics_evaluation

# Assume 'tables' is your list from the screenshot
def prepare_inputs_parent(rdfs):
    """
    Cleans the RDF pairs and transforms them into the proper format so that the PARENT module can calculate with it.
    Input: RDF pairs of format "Attribute <s> Value <p> Relation <o>"
    Returns a list of lists containing tuples --> [ [(Attribute, Relation, Value), ...] ...]
    """
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
    gen_file = os.path.join(OUTPUT_DIR, f'gen_{lang_code}.txt')
    ref_file = os.path.join(OUTPUT_DIR, f'ref_{lang_code}.txt')
    table_file = os.path.join(OUTPUT_DIR, f'table_{lang_code}.txt')

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
        "python", PARENT_SCRIPT_PATH,
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
    language_metrics = {}
    languages = ['mt','br', 'cy', 'ga']
    for lang in languages:
        # Load language-specific test data
        test_dataset = load_test_data_for_language(lang)
        test_dataset = test_dataset
        test_ds, true_articles_test, test_rdf_input = transform_datasets(test_dataset)

        # Generate predictions
        predictions = generate_predictions(model, test_ds, lang)

        # Evaluate predictions using standard metrics
        evaluation_results = evaluate_test_set(predictions, true_articles_test,lang)
        if lang != 'mt':
            language_metrics[lang] = evaluation_results
        else:
            language_metrics[lang] =  {
                    "blue": evaluation_results["bleu"],
                    "chrf": evaluation_results.get("chrf"),  # Assuming this is the key for chrF++
                    "TER": evaluation_results["TER"]["score"]
                                            }

        with open(f'{PATH}/2nd_experiment_generated_files/evaluation_results_{lang}.json', 'w') as file:
            json.dump(evaluation_results, file)

        # PARENT evaluation
        rdf_triples = prepare_inputs_parent(test_rdf_input)  # Assuming this function prepares RDF triples correctly
        parent_results = evaluate_individual_rdf(predictions, true_articles_test, rdf_triples, lang)
        write_rdfs_to_file(rdf_triples, f'{PATH}/2nd_experiment_generated_files/tables{lang}.txt')
        parent_results = json.loads(parent_results)
        #parent_results = {k: sum(v) / len(v) for k, v in parent_results.items() if v}
        combined_results = {**evaluation_results, **parent_results}

        # Now write the combined results to a language-specific file
        with open(f'{PATH}/2nd_experiment_generated_files/combined_evaluation_results_{lang}.json', 'w') as file:
            json.dump(combined_results, file)





