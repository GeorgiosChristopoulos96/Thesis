import numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EarlyStoppingCallback)
import pandas as pd
import torch
from Utils import  evaluate_texts, CUDA_CLEAN
from transformers import TrainerCallback
DATASET_PATH = "/Users/georgioschristopoulos/PycharmProjects/Thesis"
if torch.cuda.is_available():
    torch.device('cuda')
    CUDA_CLEAN()
else:
    torch.device('cpu')
train_tsv = f'{DATASET_PATH}/combined_train.tsv'
dev_tsv = f'{DATASET_PATH}/combined_dev.tsv'
test_tsv = f'{DATASET_PATH}/combined_test.tsv'
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
    return Dataset.from_pandas(df[['input_text', 'target_text', 'prefix']])

# Load the datasets
train_dataset = load_dataset_from_tsv(train_tsv)
dev_dataset = load_dataset_from_tsv(dev_tsv)
test_dataset = load_dataset_from_tsv(test_tsv)
# Define tokenizer and model
checkpoint = f"{DATASET_PATH}/finetune_RDF_to_text_epoch/checkpoint-6180"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,legacy = False, use_fast = False)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")


# Preprocessing function with prefix included
def preprocess_function(examples):
    language_prefixes = {
        'English': 'RDF-to-text in English:',
        'Russian': 'RDF-to-text in Russian:',
        'German': 'RDF-to-text in German:',
        'Italian': 'RDF-to-text in Italian:',
        'French': 'RDF-to-text in French:',
        'Arabic': 'RDF-to-text in Arabic:',
        'Hebrew': 'RDF-to-text in Hebrew:',
        'Amharic': 'RDF-to-text in Amharic:',
        'Galician': 'RDF-to-text in Galician:'

    }

    # Create the inputs using the language-specific prefixes
    inputs = [f"{language_prefixes[lang]} {example}" for example, lang in
              zip(examples["input_text"], examples["prefix"])]

    # Prepend the prefix to each input text
    # inputs = ["RDF-to-text in English: " + example for example in examples['input_text']]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length", return_tensors="pt")

    # Tokenize the targets as before
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length",
                           return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_dev_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

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
training_args = Seq2SeqTrainingArguments(
    output_dir = "./finetune_RDF_to_text_epoch_MIX_OPUS_LANGS",
    max_steps =100000,
    per_device_train_batch_size= 8,
    per_device_eval_batch_size= 8,
    learning_rate = 2e-5,
    logging_dir = "./logs",
    evaluation_strategy ="steps",
    save_strategy = "steps",
    load_best_model_at_end =True,
    eval_steps = 5000,
    logging_strategy = 'epoch',
    save_total_limit =2,
    save_steps = 5000,
    gradient_accumulation_steps = 8,
    predict_with_generate = True
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3),metrics_callback],
)
# Train and save the model
trainer.train()
trainer.save_model("./RDF_to_text_epoch_MIX_OPUS_LANGS")

log_history = trainer.state.log_history
def scale_losses_with_first_as_max(values):
    # If the first value is 0, scaling will cause division by zero.
    # In such a case, we can either skip scaling or set a minimum scale factor.
    if values[0] == 0:
        return values  # or you can return [val / (values[0] + epsilon) for val in values]
    max_loss = values[0]
    return [val / max_loss for val in values]


def extract_metrics_for_plotting(metrics_history, metric_keys):
    # Initialize a dictionary to hold the metrics
    metrics_for_plotting = {}

    # Extract the values for each metric key
    for key in metric_keys:
        metric_values = extract_metric_values(metrics_history, key)
        metrics_for_plotting[key] = metric_values

    return metrics_for_plotting
def extract_metric_values(metrics_history, metric_key):
    values = []
    for entry in metrics_history:
        # For nested metrics like 'eval_TER', extract the 'score' key
        if isinstance(entry.get(metric_key), dict):
            values.append(entry[metric_key]['score'])
        else:
            values.append(entry[metric_key])
    return values

metrics_history = metrics_callback.metrics_history

# Prepare data for plotting
steps = [entry['epoch'] for entry in log_history if 'eval_loss' in entry]  # epochs where eval_loss is recorded
metrics_dict = {
    'Training Loss': [entry['train_loss'] for entry in log_history if 'train_loss' in entry],
    'Validation Loss': [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry],
}

# Ensure that steps align with the length of metrics
# Here, we assume that `Training Loss` and `Validation Loss` are recorded at the same epochs
# Adjust accordingly if they are not
if len(steps) != len(metrics_dict['Validation Loss']):
    # Adjust steps to match the length of Validation Loss entries if they are different
    steps = steps[:len(metrics_dict['Validation Loss'])]

# Proceed with file writing and plotting
filename = f"{DATASET_PATH}/logs/metrics_val_train_loss"
with open(filename, "w") as file:
    for key, value in metrics_dict.items():
        file.write(f"{key}: {value}\n")

eval_preds = trainer.predict(tokenized_test_dataset)
filename = f"{DATASET_PATH}/logs/metrics_test"
metrics = eval_preds.metrics
# Write the metrics to the file
with open(filename, "a") as file:
    for key, value in metrics.items():
        file.write(f"{key}: {value}\n")