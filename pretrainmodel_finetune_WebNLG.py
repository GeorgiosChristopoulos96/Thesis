import json

import evaluate
import numpy as np
from datasets import Dataset, load_metric
import wandb
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
import pandas as pd
import torch
import matplotlib.pyplot as plt
from Utils import plot_dual_metrics, evaluate_texts

wandb.init(project="rdf-to-text", entity="gogot53")
with open('config_finetune.json', 'r') as f:
    config_args = json.load(f)
# Log hyperparameters (optional but recommended)
wandb.config.update({k: v for k, v in config_args.items() if k != "wandb"})
DATASET_PATH = "/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_En/release_v3.0/ALT_prep"

# Paths to your TSV files
train_tsv = f'{DATASET_PATH}/train.tsv'
dev_tsv = f'{DATASET_PATH}/dev.tsv'
test_tsv = f'{DATASET_PATH}/test.tsv'




# Function to load a TSV file and prepare a Hugging Face Dataset
def load_dataset_from_tsv(tsv_path):
    df = pd.read_csv(tsv_path, delimiter='\t')
    # Assuming 'triples' and 'lexicalization' columns exist in your TSV files
    df['input_text'] = df['triples']
    df['target_text'] = df['lexicalization']
    return Dataset.from_pandas(df[['input_text', 'target_text']])
#split each lexicalization into a separate row duplicate the triples and input_text
def load_dataset_from_tsv_single_lex(tsv_path):
    df = pd.read_csv(tsv_path, delimiter='\t')
    # Expand the 'lexicalization' column into a list of lexicalizations
    df['lexicalizations'] = df['lexicalization'].apply(lambda x: x[1:-1].split('", "'))
    # Explode the DataFrame so each lexicalization gets its own row, duplicating the triples
    df_exploded = df.explode('lexicalizations')
    # Strip leading and trailing quotes from each lexicalization
    df_exploded['lexicalizations'] = df_exploded['lexicalizations'].str.strip('"')
    #remove columns where the lexicalization is empty
    df_exploded = df_exploded[df_exploded['lexicalizations'] != '']

    # Use the triples as 'input_text' directly without the prefix
    df_exploded['input_text'] = df_exploded['triples'].apply(str.lower)
    # The 'lexicalizations' column is used as the target text
    df_exploded['target_text'] = df_exploded['lexicalizations']
    return Dataset.from_pandas(df_exploded[['input_text', 'target_text']])

# Load the datasets
train_dataset = load_dataset_from_tsv_single_lex(train_tsv).select(range(10))  # Adjust according to your RAM
dev_dataset = load_dataset_from_tsv_single_lex(dev_tsv).select(range(10))  # Adjust according to your RAM
test_dataset = load_dataset_from_tsv(test_tsv)
true_articles_dev = dev_dataset['target_text']
# Define tokenizer and model
checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,legacy = False, use_fast = False)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,dropout_rate=0.1).to("cuda" if torch.cuda.is_available() else "mps")


max_input_length = 1024
max_target_length = 512
# Preprocessing function with prefix included
def preprocess_function(examples):
    # Prepend the prefix to each input text
    inputs = ["RDF-to-en: " + example for example in examples['input_text']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    # Tokenize the targets as before
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# Define training arguments
training_args = Seq2SeqTrainingArguments(**config_args)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define metrics for evaluation
#metric = load_metric("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decoding the predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Handling -100 values for labels which are used to ignore some tokens in loss computation
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-processing the texts
    #decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    bleu_output, rouge_output, meteor_output, ter, bertscore_F1,bertscore_P, bertscore_R = evaluate_texts(decoded_preds, decoded_labels)
    # Calculate metrics

    return {"bleu": bleu_output["score"], "rouge": rouge_output["rougeL"], "meteor": meteor_output["meteor"],
            "TER": ter, "bertscore_F1": bertscore_F1, "bertscore_P": bertscore_P, "bertscore_R": bertscore_R}

# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     if isinstance(preds, tuple):
#         preds = preds[0]
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     decoded_labels = [[label] for label in decoded_labels]
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     wandb.log({"BLEU": result["score"]})
#     return {"bleu": result["score"]}

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics= compute_metrics
)

# Train and save the model
trainer.train()
#trainer.save_model("./rdf_to_text_model_final")



# Assuming 'trainer_state' is your TrainerState object with the log_history attribute filled with data
log_history = trainer.state.log_history


# Define a function to plot training and validation metrics on the same graph
# def scale_losses_with_first_as_max(values):
#     # Set the first value as the max loss
#     max_loss = values[0]
#     # Scale other values relative to the first value
#     scaled_values = [val / max_loss for val in values]
#     return scaled_values
#
# def plot_dual_metrics(steps, values1, values2, title, y_label, filename,
#                       label1='Training', label2='Validation'):
#     # Scale the values so that the first value is 1
#     values1 = scale_losses_with_first_as_max(values1)
#     values2 = scale_losses_with_first_as_max(values2)
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(steps, values1, label=label1, color='blue', marker='o')
#     plt.plot(steps, values2, label=label2, color='orange', marker='o')
#     plt.title(title)
#     plt.xlabel('Steps')
#     plt.ylabel(y_label)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close()

# Assuming log_history is a list of dictionaries with metrics logged at different steps



# Initialize lists to store the x and y values for each plot
steps = []
loss_values = []
eval_loss_values = []


# Extract the metrics
for entry in log_history:
    if 'loss' in entry:
        # Ensure 'step' is recorded only once for each entry that has 'loss'
        if 'step' in entry and (not steps or entry['step'] > steps[-1]):
            steps.append(entry['step'])
        loss_values.append(entry['loss'])

    if 'eval_loss' in entry:
        # 'eval_loss' is usually logged at the same step as 'loss', but let's handle it just in case
        if 'step' in entry and (not steps or entry['step'] > steps[-1]):
            steps.append(entry['step'])
        eval_loss_values.append(entry['eval_loss'])

plot_dual_metrics(
    steps,
    loss_values,
    eval_loss_values,
    'Training and Validation Loss',
    'Loss',
    '/Users/georgioschristopoulos/PycharmProjects/Thesis/loss_plot_finetune.png'
)
model_artifact = wandb.Artifact(
    name="rdf_to_text_model",
    type="model",
    description="RDF to text translation model trained on WebNLG dataset."
)

# Add the trained model directory to the artifact
model_artifact.add_dir("./rdf_to_text_model_final")

# Log the artifact to your wandb project
wandb.log_artifact(model_artifact)

# Finish the wandb run
wandb.finish()
