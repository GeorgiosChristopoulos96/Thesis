import json
import os
import transformers
from matplotlib import ticker
import wandb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer, EarlyStoppingCallback, Adafactor,TrainerCallback
from datasets import load_from_disk
import numpy as np
import torch
import gc
import evaluate
import nltk
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
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
class MetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.metrics_history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.metrics_history.append(metrics)



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decoding the predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Handling -100 values for labels which are used to ignore some tokens in loss computation
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_output, rouge_output, meteor_output, ter, chrf, bertscore_F1,bertscore_P, bertscore_R = evaluate_texts(decoded_preds, decoded_labels)
    filename = "/Users/georgioschristopoulos/PycharmProjects/Thesis/evaluation_steps_metrics.txt"


    metrics_evaluation = {"bleu": bleu_output["score"], "rouge": rouge_output["rougeL"], "meteor": meteor_output["meteor"],
            "TER": ter, "chrf":chrf, "bertscore_F1": bertscore_F1, "bertscore_P": bertscore_P, "bertscore_R": bertscore_R}
    with open(filename, "a") as file:
        file.write(f"Evaluation at step {trainer.state.global_step}:\n")
        for key, value in metrics_evaluation.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
    return metrics_evaluation

def preprocess_function(examples,source_lang, target_lang):
    inputs = examples[source_lang]
    targets = examples[target_lang]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True,padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length",return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

path_to_shards = "/Users/georgioschristopoulos/PycharmProjects/Thesis"

train = "/train_no_augm_dataset"
test = "/test_no_augm_dataset"
validation = "/validation_no_augm_dataset"
# Now you have the dataset loaded and can access the train, test, and validation splits normally
train_dataset = load_from_disk(f"{path_to_shards}/{train}").select(range(10))
test_dataset = load_from_disk(f"{path_to_shards}/{test}").select(range(10))
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


metrics_callback = MetricsCallback()
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3),metrics_callback],
)

# Train and save the model
trainer.train()
trainer.save_model("./mt5_model_pretrain_curves")

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

def plot_metrics_or_loss(steps, data_dict, title, ylabel, file_prefix, directory, is_loss=False, apply_scaling=False):
    plt.figure(figsize=(10, 6))

    for label, values in data_dict.items():
        if apply_scaling and is_loss:
            scaled_values = scale_losses_with_first_as_max(values)
            plt.plot(steps, scaled_values, label=label)
        else:
            plt.plot(steps, values, label=label)

    plt.title(title)
    plt.xlabel('Evaluation Steps (in thousands)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    # Define a function that formats the ticks as 'number' + 'k'
    def format_func(value, tick_number):
        # Check if the value is greater than 0 and less than the maximum step
        if 0 <= value < max(steps):
            return f'{int(value)}k' if value >= 1000 else str(int(value))
        return ''

    # Use the custom function to format the x-axis ticks
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{file_prefix}.png"))
    plt.close()  # Close the figure to avoid display issues

metrics_history = metrics_callback.metrics_history

# Prepare data for plotting
steps = [entry['step'] for entry in log_history if 'loss' in entry]
#loss_values = [entry['loss'] for entry in log_history if 'loss' in entry]
#eval_loss_values = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
metrics_dict = {
    'Training Loss': [entry['loss'] for entry in log_history if 'loss' in entry],  # Replace with your actual values
    'Validation Loss': [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry],
}
# Plot training and validation loss
plot_metrics_or_loss(
    steps,
    metrics_dict,  # This is the dictionary containing loss values
    'Training and Validation Loss',
    'Loss',
    'loss_plot',
    "/Users/georgioschristopoulos/PycharmProjects/Thesis",
    is_loss=True,  # Specify that this is loss data
    apply_scaling=True  # Specify whether to apply scaling
)
eval_preds = trainer.predict(test_dataset)
filename = "/Users/georgioschristopoulos/PycharmProjects/Thesis/metrics_test.txt"
metrics = eval_preds.metrics
# Write the metrics to the file
with open(filename, "w") as file:
    for key, value in metrics.items():
        file.write(f"{key}: {value}\n")



def plot_single_metric(steps, values, metric_name, directory, apply_scaling=False):
    plt.figure(figsize=(10, 6))
    if apply_scaling:
        scaled_values = scale_losses_with_first_as_max(values)
        plt.plot(steps, scaled_values, label=metric_name)
    else:
        plt.plot(steps, values, label=metric_name)

    plt.title(f'{metric_name.upper()} Score over Time')
    plt.xlabel('Evaluation Steps (in thousands)')
    plt.ylabel(f'{metric_name.capitalize()} Score')
    plt.legend()
    plt.grid(True)

    # Format the x-axis labels with 'k' suffix for thousand
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else int(x)))

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{metric_name}_plot.png"))
    plt.close()

# Loop over each metric and call plot_single_metric for each one



# Prepare the steps (x-axis values)
steps = list(range(len(metrics_history)))
# Define the metrics to plot (you may add or remove metrics based on your needs)
metric_keys = [
    'eval_loss', 'eval_bleu', 'eval_rouge', 'eval_meteor', 'eval_chrf',
    'eval_bertscore_F1', 'eval_bertscore_P', 'eval_bertscore_R'
]

for metric_name in metric_keys:
    metric_values = extract_metric_values(metrics_history, metric_name)
    # Check if metric_values is not empty to avoid plotting empty data
    if metric_values:
        plot_single_metric(
            steps,
            metric_values,
            metric_name,
            "/Users/georgioschristopoulos/PycharmProjects/Thesis",
            apply_scaling=False  # Change to True if scaling is needed
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