import ast

from datasets import load_dataset

from src.path import dataset_dir


# load

def load_dataset_from_csv_file(file_path):
    return load_dataset('csv', data_files=file_path, split="train")


def load_original_dataset(split: str):
    return load_dataset('csv', data_files=str(dataset_dir / split / f"{split}.csv"), split="train")


# preprocess

def preprocess_utterance(data, input_format) -> dict:
    output = {}
    for idx in [1, 2]:
        utterance = ast.literal_eval(data[f"utterance{idx}"])
        input_str = ""
        if "user" in input_format:
            input_str += utterance["user"] + ": "
        if "text" in input_format:
            input_str += utterance["text"]
        
        if "intent" in input_format:
            input_str += f" ({utterance['intent']})"
        
        output[f"utterance{idx}"] = input_str
    
    return output
