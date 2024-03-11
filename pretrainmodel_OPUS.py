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
nltk.download('wordnet')
nltk.download('punkt')
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
DATASET_PATH= "/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/test"
with open('config_pretrain.json', 'r') as f:
    config_args = json.load(f)
training_args = Seq2SeqTrainingArguments(**config_args)
wandb.init(project= "mt5-pretrained-model", entity= "gogot53",)
wandb.config.update({k: v for k, v in config_args.items() if k != "wandb"})
#MAYBE IMPLEMENT CONTINUOUS LEARNING

def ensure_cuda_compatability():
    print(f'Torch version: {torch.__version__}')
    print(f'Cuda version: {torch.version.cuda}')
    print(f'Cudnn version: {torch.backends.cudnn.version()}')
    print(f'Is cuda available: {torch.cuda.is_available()}')
    print(f'Number of cuda devices: {torch.cuda.device_count()}')
    print(f'Current default device: {torch.cuda.current_device()}')
    print(f'First cuda device: {torch.cuda.device(0)}')
    print(f'Name of the first cuda device: {torch.cuda.get_device_name(0)}\n\n')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    #Ensure we are really working with full GPU capacity
    gc.collect()
    torch.cuda.empty_cache()
#ensure_cuda_compatability()
if not torch.backends.mps.is_available():
    print("MPS not available")
    mps_device = torch.device("cpu")  # Fall back to CPU if MPS is not available
else:
    mps_device = torch.device("mps")

checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, legacy = False, use_fast = False)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(mps_device)




# Adjust the number of examples to use for training
# source_lang = "en"
# target_lang = "am"

def format_only_labels(labels):
    formatted_labels = []
    for label in labels:
        # Check if the label is already in the desired format: a list with a single string element
        if isinstance(label, list) and len(label) == 1 and isinstance(label[0], str):
            formatted_label = label
        else:
            # This block will handle cases where the label is not in the desired format,
            # including when it's a list with multiple elements or contains brackets
            if isinstance(label, list):
                # Concatenate elements, remove brackets, and strip whitespace
                formatted_label = [' '.join(label).replace('[', '').replace(']', '').strip()]
            else:
                # For single string elements, just ensure no brackets and strip
                formatted_label = [label.replace('[', '').replace(']', '').strip()]
        formatted_labels.append(formatted_label)
    return formatted_labels
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = format_only_labels(labels)
    return preds, labels
def load_eval_metrics():
    """
    Loads in all metrics that will be used later on during evaluation. This is seperated to not load in the metrics a dozen of times during training.
    """
    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    perplexity = evaluate.load("perplexity", module_type="metric")
    bertscore = evaluate.load("bertscore")

    print('LOGGING: load_eval_metrics DONE \n')

    return bleu, rouge, meteor, perplexity, bertscore
def evaluate_texts(decoded_preds, decoded_labels):
    """
    Calculates metrics given a list of decoded predictions and decoded labels
    """

    #post_process for BLEU
    blue_preds, blue_labels = postprocess_text(decoded_preds,  decoded_labels)

    # setup metrics for use
    bleu, rouge, meteor,perplexity, bertscore = load_eval_metrics()
    log_filename = '/Users/georgioschristopoulos/PycharmProjects/Thesis/logs/metrics_log.txt'
    #Calculate the metrics
    print(f'\n LOGGING: Calculating Blue')
    bleu_output = bleu.compute(predictions=blue_preds, references=blue_labels)
    print(f'\n LOGGING: Calculating Rouge')
    rouge_output = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    print(f'\n LOGGING: Calculating Meteor')
    meteor_output = meteor.compute(predictions= decoded_preds, references=decoded_labels)
    print(f'\n LOGGING: Calculating Perplexity')
    perp_output = perplexity.compute(predictions= decoded_preds, model_id='gpt2')
    print(f'\n LOGGING: Calculating Bertscore')
    bertscore_out= bertscore.compute(predictions= decoded_preds, references=decoded_labels, lang="en")
    P = bertscore_out['precision']
    R = bertscore_out['recall']
    F1 = bertscore_out['f1']
    # Convert to list if they are not and calculate the mean
    P = float(sum(list(P)) / len(P))
    R = float(sum(list(R)) / len(R))
    F1 = float(sum(list(F1)) / len(F1))

    # Log the mean BERTScores to Weights & Biases
    wandb.log({
        "BERTScore_Precision_mean": P,
        "BERTScore_Recall_mean": R,
        "BERTScore_F1_mean": F1
    })
    with open(log_filename, 'a') as log_file:
        log_file.write(f"\n LOGGING: Calculating Perplexity\n")
        # Assume perp_output is a dictionary that includes perplexity value under 'perplexity' key
        log_file.write(f"Perplexity: {perp_output}\n")

        log_file.write(f"\n LOGGING: Calculating Bertscore\n")
        log_file.write(f"Bertscore Precision Mean: {P}\n")
        log_file.write(f"Bertscore Recall Mean: {R}\n")
        log_file.write(f"Bertscore F1 Mean: {F1}\n")

    print(f'\n LOGGING: Done calculations')
    wandb.log({"BLEU": bleu_output["score"], "ROUGE": rouge_output["rougeL"], "METEOR": meteor_output["meteor"],
               "Perplexity": perp_output, "BERTScore_P": F1, "BERTScore_P": P, "BERTScore_R": R})
    return bleu_output, rouge_output, meteor_output, perp_output, F1, P, R
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
    bleu_output, rouge_output, meteor_output, perp_output, bertscore_F1,bertscore_P, bertscore_R = evaluate_texts(decoded_preds, decoded_labels)
    # Calculate metrics

    return {"bleu": bleu_output["score"], "rouge": rouge_output["rougeL"], "meteor": meteor_output["meteor"],
            "perplexity": perp_output, "bertscore_F1": bertscore_F1, "bertscore_P": bertscore_P, "bertscore_R": bertscore_R}

def preprocess_function(examples,source_lang, target_lang):
    #inputs = [prefix + example[source_lang] for example in examples["translation"]]
    inputs = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True,padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length",return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



# List all the language pairs you want to process

#language_pairs = [("en", "am"), ("en", "ar"), ("en", "fr"),("en","ga"),("en","gd"), ("en","it"),("en","he")]

# Load your dataset

# Placeholder for the tokenized datasets
# tokenized_datasets = {
#     "train": [],
#     "test": [],
#     "validation": []
# }
#
# for source_lang, target_lang in language_pairs:
#     file_path = f"{DATASET_PATH}/{source_lang}-{target_lang}"  # Adjust path as needed
#     data_files = {
#         "train": f"{file_path}/train.json",
#         "test": f"{file_path}/test.json",
#         "validation": f"{file_path}/validation.json"
#     }
#
#     # Load the dataset
#     dataset = load_dataset("json", data_files=data_files, field="translation")
#
#     # Preprocess the dataset
#     tokenized_dataset = dataset.map(
#         lambda examples: preprocess_function(examples, source_lang, target_lang),
#         batched=True
#     )
#
#     # Append the preprocessed datasets for concatenation
#     for split in tokenized_datasets:
#         tokenized_datasets[split].append(tokenized_dataset[split])
#
# # Concatenate the datasets for each split
# for split in tokenized_datasets:
#     tokenized_datasets[split] = concatenate_datasets(tokenized_datasets[split]) if tokenized_datasets[split] else None
path_to_shards = "/Users/georgioschristopoulos/PycharmProjects/Thesis"

train = "/train_dataset"
test = "/test_dataset"
validation = "/validation_dataset"
# Now you have the dataset loaded and can access the train, test, and validation splits normally
train_dataset = load_from_disk(f"{path_to_shards}/{train}")
test_dataset = load_from_disk(f"{path_to_shards}/{test}")
validation_dataset = load_from_disk(f"{path_to_shards}/{validation}")
# tokenized_data = data.map(preprocess_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

#might have to adjust learning rate
#maybe more training steps
#warm up steps can be proportonal to the number of training steps
optimizer = Adafactor(model.parameters(),lr=0.001,relative_step=False)

# Create the learning rate scheduler
scheduler = transformers.get_inverse_sqrt_schedule(
    optimizer, num_warmup_steps= 500
)
training_args = Seq2SeqTrainingArguments(**config_args)
# Setup the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    optimizers=(optimizer, scheduler),
    train_dataset=train_dataset,
    eval_dataset= validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train and save the model
trainer.train()
trainer.save_model("./mt5_model_pretrain_FULL")


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