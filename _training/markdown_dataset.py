from datasets import Dataset
import os

def load_markdown_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                texts.append(text)
    return {"text": texts}

dataset = Dataset.from_dict(load_markdown_files("./files"))
dataset.save_to_disk("markdown_dataset")
