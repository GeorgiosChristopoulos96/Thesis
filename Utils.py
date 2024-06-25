import gc
import evaluate
import pandas as pd
import torch
from matplotlib import pyplot as plt
from datasets import Dataset

PATH = "/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/WebNLG_br_mt_cy/2023-Challenge/data/ALT_prep/dev_"
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
    ter = evaluate.load('ter')
    chrf = evaluate.load("chrf")
    bertscore = evaluate.load("bertscore")

    print('LOGGING: load_eval_metrics DONE \n')

    return bleu, rouge, meteor, ter, chrf, bertscore


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


def scale_losses_with_first_as_max(values):
    # Set the first value as the max loss
    max_loss = values[0]
    # Scale other values relative to the first value
    scaled_values = [val / max_loss for val in values]
    return scaled_values

def plot_dual_metrics(steps, values1, values2, title, y_label, filename,
                      label1='Training', label2='Validation'):
    # Scale the values so that the first value is 1
    values1 = scale_losses_with_first_as_max(values1)
    values2 = scale_losses_with_first_as_max(values2)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, values1, label=label1, color='blue', marker='o')
    plt.plot(steps, values2, label=label2, color='orange', marker='o')
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def is_number(s):
    try:
        float(s)  # Test if it can be converted to a float
        return True
    except ValueError:
        return False

def load_test_data_for_language(lang):
    test_tsv_path = f'{PATH}{lang}.tsv'
    return load_dataset_from_tsv(test_tsv_path)

def load_dataset_from_tsv(tsv_path):
    df = pd.read_csv(tsv_path, delimiter='\t')
    # Assuming 'triples' and 'lexicalization' columns exist in your TSV files
    df['input_text'] = df['triples']
    df['target_text'] = df['lexicalization']
    return Dataset.from_pandas(df[['input_text', 'target_text']])

def evaluate_texts(decoded_preds, decoded_labels,lang):
    """
    Calculates metrics given a list of decoded predictions and decoded labels
    """

    #post_process for BLEU
    formatted_preds, formatted_labels = postprocess_text(decoded_preds,  decoded_labels)
    adjusted_preds = [s.replace("'", '"') for s in formatted_preds]
    adjusted_labels = [[s.replace('"', '').replace("'", '') for s in sublist] for sublist in formatted_labels]

    bleu, rouge, meteor, ter, chrf, bertscore = load_eval_metrics()

    print(f'\n LOGGING: Calculating Blue')
    bleu_output = bleu.compute(predictions=adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating Rouge')
    rouge_output = rouge.compute(predictions=adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating Meteor')
    meteor_output = meteor.compute(predictions= adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating TER')
    ter_output = ter.compute(predictions= adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating chrf')
    chrf_output = chrf.compute(predictions=adjusted_preds, references=adjusted_labels)
    print(f'\n LOGGING: Calculating Bertscore')
    bertscore_out= bertscore.compute(predictions= adjusted_preds, references=adjusted_labels, lang=lang)
    P = bertscore_out['precision']
    R = bertscore_out['recall']
    F1 = bertscore_out['f1']
    # Convert to list if they are not and calculate the mean
    P = float(sum(list(P)) / len(P))
    R = float(sum(list(R)) / len(R))
    F1 = float(sum(list(F1)) / len(F1))

    print(f'\n LOGGING: Done calculations')
    return bleu_output, rouge_output, meteor_output, ter_output, chrf_output, F1, P, R


def CUDA_CLEAN():
    print(f'Torch version: {torch.__version__}')
    print(f'Cuda version: {torch.version.cuda}')
    print(f'Cudnn version: {torch.backends.cudnn.version()}')
    print(f'Is cuda available: {torch.cuda.is_available()}')
    print(f'Number of cuda devices: {torch.cuda.device_count()}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gc.collect()
    torch.cuda.empty_cache()

