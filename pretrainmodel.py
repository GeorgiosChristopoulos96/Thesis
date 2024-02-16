from collections import Counter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, EarlyStoppingCallback
from datasets import load_dataset, load_metric
import numpy as np
import torch
import gc
import evaluate
import datasets
import nltk
nltk.download('wordnet')
nltk.download('punkt')

#MAYBE IMPLEMENT CONTINUOUS LEARNING
# torch.device("mps")
# Use mT5 model
# Check that MPS is available
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

books = load_dataset(
    'json',
    data_files={
        'train': '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/Maltese/am-en/train_with_id.json',
        'test': '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/Maltese/am-en/test_with_id.json',
        'validation': '/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/Maltese/am-en/validation_with_id.json'

    },
    field='translation'  # Specify the field containing the data records
)

# Adjust the number of examples to use for training
source_lang = "am"
target_lang = "en"
prefix = "translate amharic to english "
books["train"] = books["train"].select(range(10))
# Adjusted to English-Amharic translation
 # This can be adjusted or removed based on your preference and testing

token_counter = Counter()
# Assuming 'books' is your dataset and 'text' is the field containing the text
for example in books['train']:
    text = example['translation'][source_lang]  # Adjust this line based on your dataset structure
    # Tokenize the text and extract only the tokens, not the full tokenization output
    tokens = tokenizer.tokenize("text")
    token_counter.update(tokens)
#add new tokens to the vocabulary
######################################################EXAMPLE############################################
#tagalog tokens
new_tokens = ["ᜌ"," ᜔ᜊ"," ᜌ", "ᜒᜈ᜔"]
vocab = set(tokenizer.get_vocab().keys())
if set(new_tokens) not in vocab:
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    new_tokens = [token for token, count in token_counter.items() if token not in vocab]
test_text = "ᜌ ᜔ᜊ ᜌᜒ ᜈ"
encoded_input = tokenizer(test_text, return_tensors="pt")
decoded_output = tokenizer.decode(encoded_input["input_ids"][0])
print(f"Found {len(new_tokens)} unique tokens not in the tokenizer's vocabulary.")
############################################################################################################
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
def load_eval_metrics():
    """
    Loads in all metrics that will be used later on during evaluation. This is seperated to not load in the metrics a dozen of times during training.
    """
    bleu = datasets.load_metric("bleu")
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

    # #Calculate the metrics
    # print(f'\n LOGGING: Calculating Blue')
    # bleu_output = bleu.compute(predictions=blue_preds, references=blue_labels)
    print(f'\n LOGGING: Calculating Rouge')
    rouge_output = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    print(f'\n LOGGING: Calculating Meteor')
    meteor_output = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    print(f'\n LOGGING: Calculating Perplexity')
    perp_output = perplexity.compute(predictions=decoded_preds, model_id='gpt2')
    print(f'\n LOGGING: Calculating Bertscore')
    bertscore_output = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

    print(f'\n LOGGING: Done calculations')

    return 'bleu_outpu', rouge_output, meteor_output, perp_output, bertscore_output
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
    bleu_output, rouge_output, meteor_output, perp_output, bertscore_output = evaluate_texts(decoded_preds, decoded_labels)
    # Calculate metrics
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    # Calculate generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    # Round results for better readability
    result = {k: round(v, 4) for k, v in result.items()}
    return result



def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    #inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"].to(mps_device)
    return model_inputs

tokenized_books = books.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

metric = load_metric("sacrebleu")

# Define compute_metrics function as before

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./my_awesome_mt5_model",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Adjust batch size according to your GPU memory
    per_device_eval_batch_size=8,
    save_total_limit=3,
    num_train_epochs=12,
    predict_with_generate=True,
    gradient_accumulation_steps = 2,
    use_mps_device=True,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    save_strategy="epoch",
    generation_num_beams=5
)
#maybe use a visualisation tool
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
trainer.save_model("./my_awesome_mt5_model")