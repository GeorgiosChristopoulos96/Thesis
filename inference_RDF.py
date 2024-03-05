import pickle

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_metric, Dataset
import pandas as pd
from torch.utils.data import DataLoader


# Assuming FLAGS is a dictionary containing configuration like batch size and model path
FLAGS = {
    'saved_model_path': './rdf_to_text_model_final',
    'batch_size': 8  # Adjust according to your setup
}
def preprocess_function(examples):
    # Prepend the prefix to each input text
    inputs = ["RDF-to-English: " + example for example in examples['input_text']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # Tokenize the targets as before
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
def load_dataset_from_tsv(tsv_path):
    df = pd.read_csv(tsv_path, delimiter='\t')
    # Assuming 'triples' and 'lexicalization' columns exist in your TSV files
    df['input_text'] = df['triples'].apply(lambda x: 'RDF-to-text: ' + x.lower())
    df['target_text'] = df['lexicalization']
    return Dataset.from_pandas(df[['input_text', 'target_text']])
def get_saved_model():
    """
    Retrieves the best model and tokenizer that was saved after fine-tuning.
    """
    saved_model = AutoModelForSeq2SeqLM.from_pretrained(FLAGS['saved_model_path'], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS['saved_model_path'], local_files_only=True, add_prefix_space=True)

    return saved_model.to("cuda" if torch.cuda.is_available() else "cpu"), tokenizer

def generate_predictions(saved_model, test_set):
    ""
    encoded_inputs = test_set.remove_columns("labels")

    # set-up a dataloader to load in the tokenized test dataset
    dataloader = torch.utils.data.DataLoader(encoded_inputs,  batch_size=FLAGS['batch_size'])

    # generate text for each batch
    all_predictions = []
    for i,batch in enumerate(dataloader):
        predictions = saved_model.generate(**batch, max_new_tokens = 100, do_sample=True, num_beams = 5, top_p=0.7, repetition_penalty = 1.3)
        all_predictions.append(predictions)

    # flatten predictions
    all_predictions_flattened = [pred for preds in all_predictions for pred in preds]

    print('LOGGING: generate_predictions DONE \n')
    return all_predictions_flattened





def decode_predictions(predictions, tokenizer):
    """
    Decode the predictions made by the model
    """
    decoded_predictions = []

    for iteration, prediction in enumerate(predictions):
        decoded_predictions.append((tokenizer.decode(prediction,skip_special_tokens=True)))

    print('LOGGING: decode_predictions DONE \n')

    return decoded_predictions

# Note: You'll need to adapt `evaluate_test_set` based on your specific evaluation functions and logging mechanism.
def evaluate_test_set(predictions, test_set, tokenizer):
    """
    Evaluate the test set based on the predictions made by the model.
    """
    decoded_predictions = decode_predictions(predictions, tokenizer)
    evaluate_test_set(decoded_predictions, test_set, tokenizer)

def transform_datasets(dataset):
    """
    After loading in and creating the initial dataset, the text data is transformed, by tokenizing the input and output texts. The initial dataset is also split into train,val,test for training use.
    NOTE That the test set will not be preprocessed here yet, this will be done in a different function
    """
    test_ds = dataset
    # to use the actual articles for evaluation
    true_articles_test = test_ds['target_text']
    # The Parent Metric requires the original RDFs
    test_rdf_input = test_ds['input_text']
    ## Process the data in batches
    test_ds = test_ds.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    # transform the datasets into torch sensors, as the model will expect this format
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    print('LOGGING: transform_datasets DONE \n')
    return  test_ds, true_articles_test, test_rdf_input

# Load the test set
if __name__ == "__main__":
    model, tokenizer = get_saved_model()
    test_tsv = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_En/release_v3.0/ALT_prep/test.tsv'
    test_dataset = load_dataset_from_tsv(test_tsv)
    test_ds, true_articles_test, test_rdf_input = transform_datasets(test_dataset)
    predictions = generate_predictions(model, test_ds)
    evaluate_test_set(predictions, test_dataset, tokenizer)



