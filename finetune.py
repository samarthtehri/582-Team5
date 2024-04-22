from datasets import load_dataset
import numpy as np
import evaluate
from transformers import AutoTokenizer, T5Tokenizer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification, T5ForSequenceClassification
from transformers import TrainingArguments, Trainer 
import pandas as pd
import torch
from torch.utils.data import Dataset
import ast
import pdb
import json


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
        formatted_string = f"{user}: {text}"
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
    return tokenizer(list(data), truncation=True, padding=True, max_length=512)

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

#model_name = "google-bert/bert-large-cased"
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
if (model_name.startswith("t5")):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name, num_labels=2)    


train_file = './train.csv'
test_file = './test.csv'

train_dataset = process_file(train_file) 
test_dataset = process_file(test_file) 


metric = evaluate.load("accuracy")
metrics = [evaluate.load("precision"), evaluate.load("recall"), evaluate.load("f1")]

aggs = [["macro", 1], ["binary", 1], ["binary", 0]]
measures = ["precision", "recall", "f1"]
def compute_metrics(eval_pred):
    #pdb.set_trace()
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)["accuracy"]
    results = np.zeros((3, 3))
    for i1, agg in enumerate(aggs):
        for i2, meas in enumerate(measures):
            res = metrics[i2].compute(predictions=predictions, references=labels, average=agg[0], pos_label=agg[1])[meas]
            results[i1][i2] = res

    return {
        "accuracy": round(accuracy, 2),
        "macro-P": round(results[0][0], 2),
        "bin1-P": round(results[1][0], 2),
        "bin0-P": round(results[2][0], 2),
        "macro-R": round(results[0][1], 2),
        "binary1-R": round(results[1][1], 2),
        "binary0-R": round(results[2][1], 2),
        "macro-F1": round(results[0][2], 2),
        "binary1-F1": round(results[1][2], 2),
        "binary0-F1": round(results[2][2], 2),
    }
 

training_args = TrainingArguments(
    output_dir="test_trainer", 
    evaluation_strategy="epoch", 
    num_train_epochs=6,
    learning_rate=2e-5)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
