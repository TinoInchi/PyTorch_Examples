import pathlib
from random import Random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import os
from pathlib import Path
import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
import requests
from pathlib import Path
import torchvision
from helper_functions import plot_predictions,plot_decision_boundary


class MultiClassClassification(nn.Module):
    def __init__(self, input_features,output_features,hidden_units=8):
        '''
        Args:
            input_features (int): Number of input features of model
            output_features (int): Number of output features of model
            hidden_units (int): Number of hidden units between layers, default 8
        Return:
            None
        Error Raises:
            None
        '''
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_units = hidden_units

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features = input_features, out_features = hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features = hidden_units, out_features = hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features = hidden_units, out_features = output_features)
            )

    def forward(self,X):
        return self.linear_layer_stack(X)

def accuracy_function(y_true,y_pred):
    '''
    Args:
        y_true (torch.tensor (dim of outputfeatures of model): Training output split tensor (y)
        y_pred (torch.tensor (dim of outputfeatures of model): Prediction of model
    Return:
        acc (torch.tensor -> scalar): Accuracy from training
    Error Raises:
        None
    '''
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

def training_function(Training_input: torch.tensor):
    '''
    Args:
        Training_input (torch.tensor (dim of input_features for model)): Training input split tensor (X)
    Return:
        loss (torch.tensor -> scalar): Loss from training
        acc (torch.tensor -> scalar): Accuracy from training
    Error Raises:
        None
    '''
    model.train(True)

    y_logits = model(Training_input)
    y_pred = torch.softmax(y_logits,dim=1)
    y_pred = torch.argmax(y_pred,dim=1)

    loss = loss_function(y_logits, y_blob_train)
    acc = accuracy_function(y_true=y_blob_train,
                            y_pred = y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, acc, y_pred

def testing_function(Testing_input: torch.tensor):
    model.eval()
    with torch.inference_mode():
        test_logits = model(Testing_input)
        test_pred = torch.softmax(test_logits, dim = 1)
        test_pred = torch.argmax(test_pred, dim = 1)
        test_loss = loss_function(test_logits,y_blob_test)
        test_acc = accuracy_function(y_true=y_blob_test,
                                     y_pred=test_pred)
    return test_loss, test_acc, test_pred



NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
LEARNING_RATE = 0.001
EPOCHS = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

# make from numpy to tensors

X_blob = torch.from_numpy(X_blob).type(torch.float).to(device)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor).to(device)
#y_blob = torch.unsqueeze(y_blob, 1)

# Split datasets into training data and test data

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                         y_blob,
                                                                         test_size = 0.2,
                                                                         random_state = RANDOM_SEED,
                                                                         stratify = y_blob)

# define Model (Batches necessary)

model = MultiClassClassification(input_features=NUM_FEATURES,
                                 output_features=NUM_CLASSES,
                                 hidden_units=20).to(device)

# define Loss function

loss_function = nn.CrossEntropyLoss()

# define Optimizer

optimizer = torch.optim.Adam(params=model.parameters(),lr=LEARNING_RATE)

# Initialize array for Result Tracking

epoch_count = []
loss_values = []
accuracy_values = []
test_loss_values = []
test_accuracy_values = []


torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

for epoch in range(EPOCHS):

    # Training Loop

    loss_train, acc_train, train_pred = training_function(X_blob_train)

    loss_values.append(loss_train)
    accuracy_values.append(acc_train)

    # Testing Loop

    loss_test, acc_test, test_pred = testing_function(X_blob_test)

    test_loss_values.append(loss_test)
    test_accuracy_values.append(acc_test)

    if epoch % 10 == 9:
        print(f'Epoch: {epoch+1} \nTraining: Loss is {loss_train} Accuracy is {acc_train}\
        \nTesting: Loss is {loss_test} Accuracy is {acc_test}')

np_loss_values = np.array(torch.tensor(loss_values).numpy()) 
np_accuracy_values = np.array(torch.tensor(accuracy_values).numpy()) 

np_test_loss_values = np.array(torch.tensor(test_loss_values).numpy()) 
np_test_accuracy_values = np.array(torch.tensor(test_accuracy_values).numpy()) 



acc = sklm.accuracy_score(y_blob_train,train_pred)
precision = sklm.precision_score(y_blob_train,train_pred,average=None)
recall = sklm.recall_score(y_blob_train,train_pred,average=None)
f1 = sklm.f1_score(y_blob_train,train_pred,average=None)
cm = sklm.confusion_matrix(y_blob_train,train_pred)
print(acc)
print(precision)
print(recall)
print(f1)
print(cm)


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model,X_blob_train,y_blob_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model,X_blob_test,y_blob_test)
#plt.show()

dirname = os.path.dirname(__file__)

torch_file_name = os.path.join(dirname,'model_MultiClassClassification_Blobs.pth')

#torch.save(model.state_dict(),torch_file_name)