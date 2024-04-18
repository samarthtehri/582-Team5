import ast
import csv

from datasets import load_dataset, Dataset

from src.path import dataset_dir


# load

def load_dataset_from_csv_file(file_path: str) -> Dataset:
    return load_dataset('csv', data_files=file_path, split="train")


def load_original_dataset(split: str) -> Dataset:
    return load_dataset('csv', data_files=str(dataset_dir / split / f"{split}.csv"), split="train")


def load_csv_file_as_list(file_path: str, remove_first_row=True) -> list:
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    
    if remove_first_row:
        data = data[1:]
    
    return data


# preprocess

def preprocess_utterance(data, input_format) -> dict:
    """This function receives one instance of preprocessed Dataset"""
    
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


# write (used for data augmentation)

def dump_dataset_to_csv(data: list[list[str]], file_path: str):
    data = [["label", "category", "utterance1", "utterance2"]] + data
    
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
