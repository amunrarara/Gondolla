from datasets import load_from_disk
from transformers import BloomTokenizerFast

dataset = load_from_disk("markdown_dataset")

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-7b1")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
tokenized_dataset.save_to_disk("preprocessed_dataset")
