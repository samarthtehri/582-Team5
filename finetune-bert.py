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


file = './582-Team5/Data/train/train.csv'
file = './582-Team5/Data/test/test.csv'

train_df = pd.read_csv(file)
train_df['one'] = train_df['utterance1'].apply(format_conversation)
train_df['two'] = train_df['utterance2'].apply(format_conversation)
train_df['text'] = list(zip(train_df['one'], train_df['two']))
train_df['label'] = train_df['label'].astype(int)

#train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

test_df = pd.read_csv(file)
test_df['one'] = test_df['utterance1'].apply(format_conversation)
test_df['two'] = test_df['utterance2'].apply(format_conversation)
test_df['text'] = list(zip(test_df['one'], test_df['two']))
test_df['label'] = test_df['label'].astype(int)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)


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

train_labels = train_df.label.values
test_labels = test_df.label.values   

train_encodings = tokenize_data(train_df.text.values)
test_encodings = tokenize_data(test_df.text.values)

train_dataset = XDataset(train_encodings, train_labels)
test_dataset = XDataset(test_encodings, test_labels)

# Add labels to the tokenized data

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