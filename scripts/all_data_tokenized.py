from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

checkpoint = "google/mt5-small"  # You can choose the model size that fits your needs
tokenizer = AutoTokenizer.from_pretrained(checkpoint,  legacy = False, use_fast = True)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
train_datasets = []
validation_datasets = []
test_datasets = []
def preprocess_function(examples, source_lang, target_lang, tokenizer):
    inputs =  examples[source_lang]
    targets = examples[target_lang]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    # We don't want to pad here since the DataCollator will handle it later
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["input_token_count"] = [len(inp) for inp in model_inputs["input_ids"]]
    model_inputs["label_token_count"] = [len(label) for label in labels["input_ids"]]

    return model_inputs


# Example of how you might load and preprocess datasets for multiple PARENT_metric pairs
language_pairs = [("en", "am"), ("en", "ar"), ("en", "fr"),("en","ga"),("en","gd"), ("en","it"),("en","he")]
all_datasets = []
file = "/Users/georgioschristopoulos/PycharmProjects/Thesis/Datasets/OPUS-100/test"
for source_lang, target_lang in language_pairs:
    # Load your dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{file}/{source_lang}-{target_lang}/train.json",
            "validation": f"{file}/{source_lang}-{target_lang}/validation.json",
            "test": f"{file}/{source_lang}-{target_lang}/test.json",
        },
        field="translation"
    )

    # Preprocess the dataset
    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, source_lang, target_lang, tokenizer),
                                    batched=True)
    all_datasets.append(tokenized_dataset)

def aggregate_token_counts(datasets):
    total_input_tokens = sum([sum(ds["input_token_count"]) for ds in datasets])
    total_label_tokens = sum([sum(ds["label_token_count"]) for ds in datasets])
    return total_input_tokens, total_label_tokens

# Example usage after preprocessing:
total_input_tokens_train, total_label_tokens_train = aggregate_token_counts([ds["train"] for ds in all_datasets])
total_input_tokens_validation, total_label_tokens_validation = aggregate_token_counts([ds["validation"] for ds in all_datasets])
total_input_tokens_test, total_label_tokens_test = aggregate_token_counts([ds["test"] for ds in all_datasets])

print(f"Train dataset: {total_input_tokens_train} input tokens, {total_label_tokens_train} label tokens")
print(f"Validation dataset: {total_input_tokens_validation} input tokens, {total_label_tokens_validation} label tokens")
print(f"Test dataset: {total_input_tokens_test} input tokens, {total_label_tokens_test} label tokens")

# Concatenate all datasets for a unified training dataset
unified_dataset_train = concatenate_datasets([ds["train"] for ds in all_datasets])
unified_dataset_test = concatenate_datasets([ds["test"] for ds in all_datasets])
unified_dataset_validation = concatenate_datasets([ds["validation"] for ds in all_datasets])

unified_dataset_train.save_to_disk("/Users/georgioschristopoulos/PycharmProjects/Thesis/train_no_augm_dataset")
unified_dataset_test.save_to_disk("/Users/georgioschristopoulos/PycharmProjects/Thesis/validation_dataset")
unified_dataset_validation.save_to_disk("/Users/georgioschristopoulos/PycharmProjects/Thesis/test_dataset")


# from datasets import load_from_disk
#
# # Load the datasets
# loaded_train_dataset = load_from_disk("/path/to/save/train_dataset")
# loaded_validation_dataset = load_from_disk("/path/to/save/validation_dataset")
# loaded_test_dataset = load_from_disk("/path/to/save/test_dataset")


