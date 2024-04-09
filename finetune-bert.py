from datasets import load_dataset
import numpy as np
import evaluate
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import pandas as pd
import torch
from torch.utils.data import Dataset
import ast

def format_conversation(conversation):
    dict_strings = conversation.split('}')
    formatted_strings = []
    for dict_string in dict_strings:
        # Skip empty strings
        if dict_string.strip() == '':
            continue
        dict_string += '}'
        data_dict = ast.literal_eval(dict_string)
        user = data_dict['user']
        text = data_dict['text']
        formatted_string = f"{text}"
        formatted_strings.append(formatted_string)
    return ' '.join(formatted_strings)

class XDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Tokenize input data
def tokenize_data(data):
    return tokenizer(list(data), truncation=True, padding=True)

def process_file(file):
    df = pd.read_csv(file)
    df['one'] = df['utterance1'].apply(format_conversation)
    df['two'] = df['utterance2'].apply(format_conversation)
    df['text'] = list(zip(df['one'], df['two']))
    df['label'] = df['label'].astype(int)

    labels = df.label.values    
    encodings = tokenize_data( df.text.values)
    dataset = XDataset(encodings, labels)
    return dataset

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

train_file = './Data/train/train.csv'
test_file = './Data/test/test.csv'

train_dataset = process_file(train_file) 
test_dataset = process_file(test_file) 

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=6)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()