# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import trange

import pdb
import ast

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
    )

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

# Function to format user and text values
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


def load_file(file, val_split=None):
    df = pd.read_csv(file)
    df['one'] = df['utterance1'].apply(format_conversation)
    df['two'] = df['utterance2'].apply(format_conversation)
    text_pairs = list(zip(df['one'], df['two']))
    labels = df.label.values

    encoding_dict = tokenizer.batch_encode_plus(
            text_pairs,
            add_special_tokens = True,
            padding=True, 
            return_attention_mask = True,
            return_tensors = 'pt'
    )

    token_ids = encoding_dict['input_ids']
    attention_masks = encoding_dict['attention_mask']
    labels = torch.LongTensor(labels)   

    if(val_split is None):
        test_set = TensorDataset(token_ids, 
                        attention_masks, 
                        labels)
        test_loader = DataLoader(test_set, batch_size=16)
        return test_loader
    
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size = val_split,
        shuffle = True,
        stratify = labels)

        # Train and validation sets
    train_set = TensorDataset(token_ids[train_idx], 
                            attention_masks[train_idx], 
                            labels[train_idx])

    val_set = TensorDataset(token_ids[val_idx], 
                            attention_masks[val_idx], 
                            labels[val_idx])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    validation_loader = DataLoader(val_set, batch_size=16)
    return train_loader, validation_loader

file = 'Data/train/train.csv'
train_loader, validation_loader = load_file(file, val_split=.1)


criterion = nn.CrossEntropyLoss()  
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def b_metrics(preds, labels):
    '''
    Returns the following metrics:
        - accuracy    = (TP + TN) / N
        - precision   = TP / (TP + FP)
        - recall      = TP / (TP + FN)
        - specificity = TN / (TN + FP)
    '''
    def b_tp(preds, labels):
        '''Returns True Positives (TP): count of correct predictions of actual class 1'''
        return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

    def b_fp(preds, labels):
        '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
        return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

    def b_tn(preds, labels):
        '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
        return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

    def b_fn(preds, labels):
        '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
        return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])
    preds = np.argmax(preds, axis = 1).flatten()
    labels = labels.flatten()
    tp = b_tp(preds, labels)
    tn = b_tn(preds, labels)
    fp = b_fp(preds, labels)
    fn = b_fn(preds, labels)
    b_accuracy = (tp + tn) / len(labels)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
    return b_accuracy, b_precision, b_recall, b_specificity

# Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
epochs = 10

for epoch in trange(epochs, desc = 'Epoch'):
    
    # ========== Training ==========
    model.train()
    
    # Tracking variables
    epoch_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, labels = batch
        optimizer.zero_grad()
        # Forward pass
        logits = model(input_ids, token_type_ids = None, attention_mask = input_mask)
        # Backward pass
        loss = criterion(logits.logits, labels)
        
        optimizer.step()
        # Update tracking variables
        epoch_loss += loss.item()
    print(f"epoch {epoch}: loss = {epoch_loss / len(train_loader)}")
    # ========== Validation ==========

    # Set model to evaluation mode
    model.eval()

    # Tracking variables 
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []

    for batch in validation_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
          # Forward pass
          eval_output = model(b_input_ids, 
                              token_type_ids = None, 
                              attention_mask = b_input_mask)
        logits = eval_output.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate validation metrics
        b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
        val_accuracy.append(b_accuracy)
        if b_precision != 'nan': val_precision.append(b_precision)
        if b_recall != 'nan': val_recall.append(b_recall)
        if b_specificity != 'nan': val_specificity.append(b_specificity)

    
    print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
    print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
    print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
    print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')


test_file = 'Data/test/test.csv'
test_loader = load_file(test_file)

test_accuracy = []
test_precision = []
test_recall = []
test_specificity = []

# Forward pass, calculate logit predictions
for batch in test_loader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        # Forward pass
        eval_output = model(b_input_ids, 
                            token_type_ids = None, 
                            attention_mask = b_input_mask)
    logits = eval_output.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    # Calculate validation metrics
    b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
    test_accuracy.append(b_accuracy)
    # Update precision only when (tp + fp) !=0; ignore nan
    if b_precision != 'nan': test_precision.append(b_precision)
    # Update recall only when (tp + fn) !=0; ignore nan
    if b_recall != 'nan': test_recall.append(b_recall)
    # Update specificity only when (tn + fp) !=0; ignore nan
    if b_specificity != 'nan': test_specificity.append(b_specificity)

print('\t - Test Accuracy: {:.4f}'.format(sum(test_accuracy)/len(test_accuracy)))
print('\t - Test Precision: {:.4f}'.format(sum(test_precision)/len(test_precision)) if len(test_precision)>0 else '\t - Validation Precision: NaN')
print('\t - Test Recall: {:.4f}'.format(sum(test_recall)/len(test_recall)) if len(test_recall)>0 else '\t - Validation Recall: NaN')
print('\t - Test Specificity: {:.4f}\n'.format(sum(test_specificity)/len(test_specificity)) if len(test_specificity)>0 else '\t - Validation Specificity: NaN')
