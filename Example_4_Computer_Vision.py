import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
import requests
from helper_functions import plot_predictions,plot_decision_boundary
from timeit import default_timer as timer
from tqdm.auto import tqdm


class CNN(nn.Module):
    def __init__(self,input_tensor: int,
                 hidden_tensor: int,
                 output_tensor: int):
        super().__init__()
        self.input_tensor = input_tensor
        self.hidden_tensor = hidden_tensor
        self.output_tensor = output_tensor

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = input_tensor,
                                             out_channels =hidden_tensor,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=hidden_tensor,
                                             out_channels=hidden_tensor,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = hidden_tensor,
                                             out_channels =hidden_tensor,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=hidden_tensor,
                                             out_channels=hidden_tensor,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))

        self.classifier = nn.Sequential(nn.Flatten(), # build a vector --> 28x28 = 784
                                         nn.Linear(in_features = hidden_tensor*7*7,
                                                   out_features = output_tensor))

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.classifier(X)

        return X


class LinearMCC(nn.Module):
    def __init__(self,input_tensor: int,
                 hidden_tensor: int,
                 output_tensor: int):
        super().__init__()
        self.input_tensor = input_tensor
        self.hidden_tensor = hidden_tensor
        self.output_tensor = output_tensor

        self.layer_stack = nn.Sequential(nn.Flatten(), # build a vector --> 28x28 = 784
                                         nn.Linear(in_features = input_tensor*7*7,
                                                   out_features = hidden_tensor))

    def forward(self, X):

        return self.layer_stack(X)


def print_train_time(start: float,
                        end: float,
                        device: torch.device = None):
    total_time = end-start
    print(f"Train time on {device}: {total_time:.3f} seconds")

# define constants

EPOCHS = 5
NUM_CLASSES = 4 # outputs
NUM_FEATURES = 2 # inputs
RANDOM_SEED = 42
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# get data and sort data

dir_name = os.path.dirname(__file__) 
data_path = os.path.join(dir_name,'data')


train_data = datasets.FashionMNIST(root = data_path,
                                    train = True,
                                    download = False,
                                    transform = ToTensor(),
                                    target_transform = None)

test_data = datasets.FashionMNIST(root = data_path,
                                    train = False,
                                    download = False,
                                    transform = ToTensor(),
                                    target_transform = None)


# split data into batches

train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH_SIZE,
                              shuffle = True)

test_dataloader = DataLoader(dataset = test_data,
                              batch_size = BATCH_SIZE,
                              shuffle = False)

class_names = train_data.classes
#class_to_idx = train_data.class_to_idx

#train_features_batch, train_labels_batch = next(iter(train_dataloader))


torch.manual_seed(RANDOM_SEED)

model = CNN(input_tensor=1,
            hidden_tensor=10,
            output_tensor=len(class_names)).to('cpu')

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=model.parameters(),lr=LEARNING_RATE)


train_time_start_cpu = timer()


for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}\n------")
    train_loss = 0
    for batch, (X,y) in enumerate(train_dataloader):
        model.train(True)

        y_pred = model(X)

        loss = loss_function(y_pred,y)
        train_loss += loss
        


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch%400 == 0:
            print(f"looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dataloader:
            test_pred = model(X_test)
            test_acc += sklm.accuracy_score(y_test,test_pred.argmax(dim=1))
            test_loss += loss_function(test_pred,y_test)

        test_loss /= len(test_dataloader)

        test_acc /= len(test_dataloader)


    print(f"\nTrain loss: {train_loss:.4f} | test loss: {test_loss:.4f}, Test acc: {100*test_acc:.4f}")
    train_time_end_cpu = timer()
    total_train_time_model = print_train_time(start=train_time_start_cpu,
                                                end=train_time_end_cpu,
                                                device=str(next(model.parameters()).device))


y_preds = []
y_tests = []
model.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader, desc='making predictions...'):
        y_logits = model(X)
        y_pred = torch.softmax(y_logits.squeeze(),dim=0).argmax(dim=1)
        y_preds.append(y_pred)
        y_tests.append(y)

y_pred_tensor = torch.cat(y_preds)
y_pred_np = y_pred_tensor.detach().numpy()

y_tests_tensor = torch.cat(y_tests)
y_test_np = y_tests_tensor.detach().numpy()

cm = sklm.confusion_matrix(y_test_np,y_pred_np)

print(cm)




