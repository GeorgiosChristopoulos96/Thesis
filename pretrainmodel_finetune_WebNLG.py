import json

import evaluate
import numpy as np
from datasets import Dataset, load_metric
import wandb
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EarlyStoppingCallback)
import pandas as pd
import torch
import matplotlib.pyplot as plt
from Utils import plot_dual_metrics, evaluate_texts
from transformers import TrainerCallback
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
class MetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.metrics_history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.metrics_history.append(metrics)






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
    bleu_output, rouge_output, meteor_output, ter, chrf, bertscore_F1,bertscore_P, bertscore_R = evaluate_texts(decoded_preds, decoded_labels,"en")
    # Calculate metrics
    filename = f"{DATASET_PATH}/evaluation_steps_metrics_finetuning.txt"
    # Use 'a' mode for appending instead of 'w' mode which overwrites
    metrics_evaluation = {"bleu": bleu_output["score"], "rouge": rouge_output["rougeL"], "meteor": meteor_output["meteor"],
            "TER": ter, "chrf":chrf, "bertscore_F1": bertscore_F1, "bertscore_P": bertscore_P, "bertscore_R": bertscore_R}
    with open(filename, "a") as file:
        file.write(f"Evaluation at step {trainer.state.global_step}:\n")
        for key, value in metrics_evaluation.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
    return metrics_evaluation


metrics_callback = MetricsCallback()
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics= compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3),metrics_callback],
)

# Train and save the model
trainer.train()
#trainer.save_model("./rdf_to_text_model_final")



# Assuming 'trainer_state' is your TrainerState object with the log_history attribute filled with data
log_history = trainer.state.log_history




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
