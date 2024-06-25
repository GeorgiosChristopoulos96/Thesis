from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PATH = "/Users/georgioschristopoulos/PycharmProjects/Thesis"
checkpoint = "google/mt5-small"  # You can choose the model size that fits your needs
tokenizer = AutoTokenizer.from_pretrained(checkpoint,  legacy = False, use_fast = False)
  # List all the language codes you have

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
train_datasets = []
validation_datasets = []
test_datasets = []
def preprocess_function(examples, source_lang, target_lang, tokenizer):
    inputs =  [f"{source_lang}-{target_lang}: " + example for example in examples[source_lang]]
    targets = [f"{source_lang}-{target_lang}: " + example for example in examples[target_lang]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    # We don't want to pad here since the DataCollator will handle it later
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# # Example of how you might load and preprocess datasets for multiple PARENT_metric pairs
language_pairs = [("en", "am"), ("en", "ar"), ("en", "fr"),("en","ga"),("en","gd"), ("en","it"),("en","he")]
all_datasets = []
file = f"{PATH}/Datasets/OPUS-100/test"
for source_lang, target_lang in language_pairs:
    # Load your dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{file}/{source_lang}-{target_lang}/cleaned_train.json",
            "validation": f"{file}/{source_lang}-{target_lang}/cleaned_validation.json",
            "test": f"{file}/{source_lang}-{target_lang}/cleaned_test.json",
        },
        field="translation"
    )

    # Preprocess the dataset
    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, source_lang, target_lang, tokenizer),
                                    batched=True)
    all_datasets.append(tokenized_dataset)


# Concatenate all datasets for a unified training dataset
unified_dataset_train = concatenate_datasets([ds["train"] for ds in all_datasets])
unified_dataset_test = concatenate_datasets([ds["test"] for ds in all_datasets])
unified_dataset_validation = concatenate_datasets([ds["validation"] for ds in all_datasets])
#
unified_dataset_train.save_to_disk(f"{PATH}/train_no_augm_dataset")
unified_dataset_test.save_to_disk(f"{PATH}/test_no_augm_dataset")
unified_dataset_validation.save_to_disk(f"{PATH}/validation_no_augm_dataset")




