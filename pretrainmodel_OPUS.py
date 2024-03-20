import json
import os

import transformers
import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, EarlyStoppingCallback, Adafactor
from datasets import load_from_disk
import numpy as np
import torch
import gc
import evaluate
import nltk

from Utils import format_only_labels, load_eval_metrics, postprocess_text, evaluate_texts, plot_dual_metrics

#nltk.download('wordnet')
#nltk.download('punkt')
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
DATASET_PATH= "/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/test"
with open('config_pretrain.json', 'r') as f:
    config_args = json.load(f)
wandb.init(project= "mt5-pretrained-model", entity= "gogot53",)
wandb.config.update({k: v for k, v in config_args.items() if k != "wandb"})
#MAYBE IMPLEMENT CONTINUOUS LEARNING


#ensure_cuda_compatability()
if not torch.backends.mps.is_available():
    print("MPS not available")
    mps_device = torch.device("cpu")  # Fall back to CPU if MPS is not available
else:
    mps_device = torch.device("mps")


checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, legacy = False, use_fast = False)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(mps_device)

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
#     return bleu, rouge, meteor, ter, bertscore

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

def preprocess_function(examples,source_lang, target_lang):
    #inputs = [prefix + example[source_lang] for example in examples["translation"]]
    inputs = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True,padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length",return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




path_to_shards = "/Users/georgioschristopoulos/PycharmProjects/Thesis"

train = "/train_dataset"
test = "/test_dataset"
validation = "/validation_dataset"
# Now you have the dataset loaded and can access the train, test, and validation splits normally
train_dataset = load_from_disk(f"{path_to_shards}/{train}").select(range(10))
test_dataset = load_from_disk(f"{path_to_shards}/{test}")
validation_dataset = load_from_disk(f"{path_to_shards}/{validation}").select(range(10))
# tokenized_data = data.map(preprocess_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

#might have to adjust learning rate
#maybe more training steps
#warm up steps can be proportonal to the number of training steps
optimizer = Adafactor(model.parameters(),lr=0.001,relative_step= False )#lr=0.001
scheduler = None
# Create the learning rate scheduler
# scheduler = transformers.get_inverse_sqrt_schedule(
#     optimizer, num_warmup_steps= 1
# )
training_args = Seq2SeqTrainingArguments(**config_args)
# Setup the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    optimizers= (optimizer,scheduler),
    train_dataset=train_dataset,
    eval_dataset= validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train and save the model
trainer.train()
#trainer.save_model("./mt5_model_pretrain_curves")

import matplotlib.pyplot as plt

# Assuming 'trainer_state' is your TrainerState object with the log_history attribute filled with data
log_history = trainer.state.log_history

#
# # Define a function to plot training and validation metrics on the same graph
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
#     plt.xlabel('Epochs')
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
    '/Users/georgioschristopoulos/PycharmProjects/Thesis/loss_plot.png'
)

model_artifact = wandb.Artifact(
    name="pretrain_text_model",
    type="model",
    description="Pretraining mT5 on OPUS."
)

# Add the trained model directory to the artifact
model_artifact.add_dir("./mt5_model_pretrain_FULL_APPLE")

# Log the artifact to your wandb project
wandb.log_artifact(model_artifact)

# Finish the wandb run
wandb.finish()