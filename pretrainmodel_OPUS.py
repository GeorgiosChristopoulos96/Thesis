from collections import Counter
import re

import transformers
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, EarlyStoppingCallback, Adafactor
from datasets import load_dataset, load_metric
import numpy as np
import torch
import gc
import evaluate
import datasets
import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
wandb.init(project="en-am-pretraining-mt5", entity="gogot53")
wandb.config.update({
    "evaluation_strategy": "epoch",  # Evaluation is done at the end of each epoch
    "learning_rate": 0.001,  # Updated to match the training arguments
    "per_device_train_batch_size": 8,  # Matches the training arguments
    "per_device_eval_batch_size": 8,  # Matches the training arguments
    "save_total_limit": 5,  # Updated to match the training arguments
    "num_train_epochs": 1,  # Updated to match the training arguments
    "predict_with_generate": True,  # Ensures predictions are generated during evaluation
    "gradient_accumulation_steps": 2,  # Matches the training arguments
    "use_mps_device": True,  # Using Apple's Metal Performance Shaders (MPS) if available
    "load_best_model_at_end": True,  # Updated to match the training arguments
    "save_strategy": "epoch",  # Model is saved at the end of each epoch
    "max_steps": 25000,  # Maximum number of training steps to perform
    "warmup_steps": 500,  # Number of warmup steps for learning rate scheduler
    "logging_steps": 1000,  # Frequency of logging training information
    "save_steps": 1000  # Frequency of saving the model
})
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

# Assuming your dataset is loaded correctly
# Here's an example of how you might load your custom dataset

data = load_dataset(
    "json",
    data_files={
        "train": "/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/Maltese/en-am/train.json",
        "test":"/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/Maltese/en-am/test.json",
        "dev": "/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/Maltese/en-am/validation.json"

    },
    field="translation"  # Specify the field containing the data records
)

# Adjust the number of examples to use for training
source_lang = "en"
target_lang = "am"
# prefix = "<Translate Amharic to English>: "
#data["train"] = data["train"].select(range(10))
# Adjusted to English-Amharic translation
 # This can be adjusted or removed based on your preference and testing

# token_counter = Counter()
# # Assuming 'books' is your dataset and 'text' is the field containing the text
# for example in books['train']:
#     text = example['translation'][source_lang]  # Adjust this line based on your dataset structure
#     # Tokenize the text and extract only the tokens, not the full tokenization output
#     tokens = tokenizer.tokenize(text)
#     token_counter.update(tokens)
#add new tokens to the vocabulary
######################################################EXAMPLE############################################
#tagalog tokens
# new_tokens = ["ᜌ"," ᜔ᜊ"," ᜌ", "ᜒᜈ᜔"]
# vocab = set(tokenizer.get_vocab().keys())
# if set(new_tokens) not in vocab:
#     tokenizer.add_tokens(new_tokens)
#     model.resize_token_embeddings(len(tokenizer))
#     new_tokens = [token for token, count in token_counter.items() if token not in vocab]
# test_text = "ᜌ ᜔ᜊ ᜌᜒ ᜈ"
# encoded_input = tokenizer(test_text, return_tensors="pt")
# decoded_output = tokenizer.decode(encoded_input["input_ids"][0])
# print(f"Found {len(new_tokens)} unique tokens not in the tokenizer's vocabulary.")
############################################################################################################
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

def preprocess_function(examples):
    #inputs = [prefix + example[source_lang] for example in examples["translation"]]
    inputs = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True,padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length",return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    # all_processed_examples = []
    # for target_language in target_lang:
    #     # Extract the inputs and targets for the current target language
    #     inputs = examples[source_lang]
    #     targets = examples[target_language]
    #
    #     # Tokenize the inputs
    #     model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
    #
    #     # Tokenize the targets
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
    #
    #     # Add the labels to model inputs
    #     model_inputs["labels"] = labels["input_ids"]
    #
    #     # Append the processed inputs for the current target language to the overall list
    #     all_processed_examples.append(model_inputs)
    #
    # return all_processed_examples
# def preprocess_function(examples, source_lang, target_lang):
#     # Initialize lists to hold processed inputs and labels
#     inputs, labels = [], []
#
#     for src_text, tgt_text in zip(examples[source_lang], examples.get(target_lang, [])):
#         # Check if the target translation exists
#         if tgt_text:
#             # Tokenize the inputs and targets
#             model_inputs = tokenizer(src_text, max_length=256, truncation=True, padding="max_length",
#                                      return_tensors="pt")
#             with tokenizer.as_target_tokenizer():
#                 labels_inputs = tokenizer(tgt_text, max_length=256, truncation=True, padding="max_length",
#                                           return_tensors="pt")
#
#             inputs.append(model_inputs)
#             labels.append(labels_inputs["input_ids"])
#         else:
#             # Handle cases where the target translation is missing
#             # For example, by appending None or using a placeholder
#             inputs.append(None)
#             labels.append(None)
#
#     # Filter out None values if you appended them in the case of missing translations
#     inputs = [i for i in inputs if i is not None]
#     labels = [l for l in labels if l is not None]
#
#     return {"input_ids": inputs, "labels": labels}
#
#
#
#
# for lang in target_lang:
#     # Filter dataset for examples where the target language translation exists
#     filtered_data = data.filter(lambda example: example.get(lang) is not None)
#
#     # Map the filtered dataset using the preprocess function
#     tokenized_data = filtered_data.map(
#         lambda examples: preprocess_function(examples, source_lang="en", target_lang=lang), batched=True)

    # Now, tokenized_data should be correctly structured for training
# def preprocess_function(examples, tokenizer, source_lang, target_lang):
#     # Initialize containers for processed data
#     processed_batch = {"input_ids": [], "attention_mask": [], "labels": []}
#
#     # Iterate over each example in the batch
#     for src_text, tgt_text in zip(examples[source_lang], examples.get(target_lang, [])):
#         # Ensure tgt_text is not None or empty
#         if tgt_text:
#             # Tokenize the source text
#             tokenized_input = tokenizer(src_text, padding="max_length", truncation=True, max_length=128)
#             # Tokenize the target text
#             tokenized_label = tokenizer(tgt_text, padding="max_length", truncation=True, max_length=128)
#
#             # Append tokenized results to the batch
#             processed_batch["input_ids"].append(tokenized_input["input_ids"])
#             processed_batch["attention_mask"].append(tokenized_input["attention_mask"])
#             processed_batch["labels"].append(
#                 tokenized_label["input_ids"])  # Assuming labels are input_ids for simplicity
#
#     return processed_batch
#
#
# from datasets import load_dataset
#
# # Assuming `data` is your dataset and `tokenizer` is already defined
#   # Example target languages
#
# for lang in target_lang:
#     # Filter to include only entries with target language translation
#     filtered_data = data.filter(lambda example: lang in example and example[lang])
#
#     # Apply preprocessing and tokenization
#     tokenized_data = filtered_data.map(
#         lambda batch: preprocess_function(batch, tokenizer=tokenizer, source_lang="en", target_lang=lang),
#         batched=True,
#         load_from_cache_file=True
#     )
# tokenized_data.save_to_disk("/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/cache")

tokenized_data = data.map(preprocess_function, batched=True)
#tokenized_data = tokenized_data.remove_columns(data["train"].column_names)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define compute_metrics function as before

# # Training arguments
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./my_awesome_mt5_model_with_beams",
#     evaluation_strategy="epoch",
#     logging_dir="./logs",
#     learning_rate=3e-4, #1e-4 and 3e-4
#     per_device_train_batch_size=8,  # Adjust batch size according to your GPU memory
#     per_device_eval_batch_size=8,
#     save_total_limit=3,
#     num_train_epochs=3,
#     predict_with_generate=True,
#     gradient_accumulation_steps = 2,
#     use_mps_device=True,
#     load_best_model_at_end=True,
#     save_strategy="epoch",
#     generation_num_beams=5,
#     report_to="wandb"
# )
# #maybe use a visualisation tool
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_data["train"],
#     eval_dataset=tokenized_data["dev"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
# )
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_pretrained_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # max_steps=25000,
    learning_rate=0.01,
    # warmup_steps=500,
    logging_dir="./logs",
    predict_with_generate=True,
    gradient_accumulation_steps=4,
    save_strategy="steps",
    evaluation_strategy="steps",
    use_mps_device=True,
    eval_steps=1000,
    logging_steps=1000,
    save_steps=1000,
    save_total_limit=5,
    load_best_model_at_end=True,
    report_to="wandb"


)

# Define the custom learning rate scheduler
# Since the original script didn't correctly implement the inverse square root schedule,
# here's a simplified example of how it might be implemented.
# Note: This is conceptual and requires adjustment based on actual use.

# def get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
#     def lr_lambda(current_step):
#         # Linear warmup
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         # Inverse square root decay
#         else:
#             return (num_warmup_steps ** 0.5) * ((current_step - num_warmup_steps) ** -0.5)
#
#     return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)



# optimizer = Adafactor(model.parameters(),lr = 0.01,relative_step=False)
# scheduler = transformers.get_inverse_sqrt_schedule(optimizer, num_warmup_steps=500)
# Define optimizer with the learning rate from the paper
optimizer = Adafactor(model.parameters(),lr=0.01,relative_step=False)

# Define the number of warmup steps (adjust based on your training data size)
num_warmup_steps = 500  # Adjust as needed


# Create the learning rate scheduler
scheduler = transformers.get_inverse_sqrt_schedule(
    optimizer, num_warmup_steps=num_warmup_steps
)
# Setup the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    optimizers=(optimizer, scheduler),
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train and save the model
trainer.train()
trainer.save_model("./mt5_model_pretrain_en_am")


model_artifact = wandb.Artifact(
    name="pretrain_text_model",
    type="model",
    description="Pretraining mT5 on OPUS."
)

# Add the trained model directory to the artifact
model_artifact.add_dir("./mt5_model_pretrain_en_am")

# Log the artifact to your wandb project
wandb.log_artifact(model_artifact)

# Finish the wandb run
wandb.finish()