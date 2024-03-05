from datasets import Dataset, DatasetDict, load_metric
import wandb
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
import pandas as pd
import torch
wandb.init(project="rdf-to-text", entity="gogot53")

# Log hyperparameters (optional but recommended)
wandb.config.update({
    "evaluation_strategy": "epoch",
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "num_train_epochs": 5,
    "predict_with_generate": True,
    "save_strategy": "epoch",
    "save_total_limit": 1
})
# Paths to your TSV files
train_tsv = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_En/release_v3.0/ALT_prep/train.tsv'
dev_tsv = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_En/release_v3.0/ALT_prep/dev.tsv'
test_tsv = '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_En/release_v3.0/ALT_prep/test.tsv'

# Function to load a TSV file and prepare a Hugging Face Dataset
def load_dataset_from_tsv(tsv_path):
    df = pd.read_csv(tsv_path, delimiter='\t')
    # Assuming 'triples' and 'lexicalization' columns exist in your TSV files
    df['input_text'] = df['triples'].apply(lambda x: 'RDF-to-text: ' + x.lower())
    df['target_text'] = df['lexicalization']
    return Dataset.from_pandas(df[['input_text', 'target_text']])
#split each lexicalization into a separate row duplicate the triples and input_text
def load_dataset_from_tsv_single_lex(tsv_path):
    df = pd.read_csv(tsv_path, delimiter='\t')
    # Expand the 'lexicalization' column into a list of lexicalizations
    df['lexicalizations'] = df['lexicalization'].apply(lambda x: x.split('","'))
    # Explode the DataFrame so each lexicalization gets its own row
    df_exploded = df.explode('lexicalizations')
    df_exploded['input_text'] = df_exploded['triples'].apply(lambda x: 'RDF-to-text: ' + x.lower())
    # Strip leading and trailing quotes from each lexicalization
    df_exploded['target_text'] = df_exploded['lexicalizations'].str.strip('"')
    return Dataset.from_pandas(df_exploded[['input_text', 'target_text']])

# Load the datasets
train_dataset = load_dataset_from_tsv(train_tsv)  # Adjust according to your RAM
dev_dataset = load_dataset_from_tsv(dev_tsv)
test_dataset = load_dataset_from_tsv(test_tsv)

# Define tokenizer and model
checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda" if torch.cuda.is_available() else "mps")

# Preprocessing function with prefix included
def preprocess_function(examples):
    # Prepend the prefix to each input text
    inputs = ["RDF-to-English: " + example for example in examples['input_text']]
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
training_args = Seq2SeqTrainingArguments(
    output_dir="./rdf_to_text_model",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,  # Adjust according to your GPU
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    predict_with_generate=True,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="wandb",
    use_mps_device=True,
    run_name="rdf_to_text_experiment",
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define metrics for evaluation
metric = load_metric("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[label] for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    wandb.log({"BLEU": result["score"]})
    return {"bleu": result["score"]}

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train and save the model
trainer.train()
trainer.save_model("./rdf_to_text_model_final")


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
