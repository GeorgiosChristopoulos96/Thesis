from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

checkpoint = "google/mt5-small"  # You can choose the model size that fits your needs
tokenizer = AutoTokenizer.from_pretrained(checkpoint,  legacy = False, use_fast = False)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
train_datasets = []
validation_datasets = []
test_datasets = []
def preprocess_function(examples, source_lang, target_lang, tokenizer):
    prefix = f"translate {source_lang} to {target_lang}: "
    inputs = [prefix + ex for ex in examples[source_lang]]
    targets = examples[target_lang]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    # We don't want to pad here since the DataCollator will handle it later
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Example of how you might load and preprocess datasets for multiple language pairs
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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    all_datasets.append(tokenized_dataset)

# Concatenate all datasets for a unified training dataset
unified_dataset_train = concatenate_datasets([ds["train"] for ds in all_datasets])
unified_dataset_test = concatenate_datasets([ds["test"] for ds in all_datasets])
unified_dataset_validation = concatenate_datasets([ds["validation"] for ds in all_datasets])

unified_dataset_train.save_to_disk("/Users/georgioschristopoulos/PycharmProjects/Thesis/train_dataset")
unified_dataset_test.save_to_disk("/Users/georgioschristopoulos/PycharmProjects/Thesis/validation_dataset")
unified_dataset_validation.save_to_disk("/Users/georgioschristopoulos/PycharmProjects/Thesis/test_dataset")


# from datasets import load_from_disk
#
# # Load the datasets
# loaded_train_dataset = load_from_disk("/path/to/save/train_dataset")
# loaded_validation_dataset = load_from_disk("/path/to/save/validation_dataset")
# loaded_test_dataset = load_from_disk("/path/to/save/test_dataset")


