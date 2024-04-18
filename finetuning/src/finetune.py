import os
from pathlib import Path
import json
import random

from tap import Tap
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, PreTrainedTokenizer

from src.utils.get_performance import get_performance
from utils.dataset_io import load_dataset_from_csv_file
from src.utils.dataset_io import preprocess_utterance


class UtteranceDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.texts.items()}
        item["label"] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)


def get_dataset(file_path, tokenizer: PreTrainedTokenizer, utterance_format=["user", "text"]):
    original_data = load_dataset_from_csv_file(file_path)
    
    texts = []
    labels = []
    for d in original_data:
        utterances = preprocess_utterance(d, utterance_format)
        input_text = f"{utterances['utterance1']}\n{utterances['utterance2']}"
        texts.append(input_text)
        
        labels.append(int(d["label"]))
    
    tokenized = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    dataset = UtteranceDataset(texts=tokenized, labels=torch.IntTensor(labels))
    return dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1).tolist()
    labels = labels.tolist()
    return get_performance(labels, predictions)
 
 
class FinetuneTap(Tap):
    model_name: str = "google-bert/bert-base-cased"
    train_file: str = './Data/train/train.csv'
    test_file: str = './Data/test/test.csv'
    utterance_format: list[str] = ["user", "text"]
    random_seed: int = 46

 
if __name__ == "__main__":
    args = FinetuneTap().parse_args()
    
    # make training reproducible
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("load tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    print("load datasets")
    train_dataset = get_dataset(args.train_file, tokenizer=tokenizer, utterance_format=args.utterance_format)
    test_dataset = get_dataset(args.test_file, tokenizer=tokenizer, utterance_format=args.utterance_format)
    
    output_dir = Path("finetuning/results") / f"model={args.model_name.split('/')[-1]},train={args.train_file.split('/')[-1]},input_format={args.utterance_format}"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(output_dir=output_dir,
                                      evaluation_strategy="epoch", num_train_epochs=3,
                                      per_device_train_batch_size=8, learning_rate=5e-5)
    
    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    
    # performance
    evaluation = trainer.evaluate()
    with open(output_dir / "performance.json", "w") as f:
        json.dump(evaluation, f, indent=4)
