from datasets import load_dataset

from src.path import dataset_dir


def load_dataset_from_csv_file(file_path):
    return load_dataset('csv', data_files=file_path, split="train")


def load_csv_dataset(split: str):
    return load_dataset('csv', data_files=str(dataset_dir / split / f"{split}.csv"), split="train")
