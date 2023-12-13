import pathlib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import os
from pathlib import Path
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path
import torchvision

class CircleModel_0(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5,out_features=1)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x))




def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

n_samples = 1000

X, y = make_circles(n_samples, noise=0.03,random_state=42)

# make DataFrame of circle data

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                     y,
                                                     test_size= 0.2,
                                                     random_state = 42,
                                                     stratify = y)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model_0 = CircleModel_0().to(device)

model_0 = nn.Sequential(nn.Linear(in_features=2,out_features=5),
                        nn.ReLU(),
                        nn.Linear(in_features=5,out_features=5),
                        nn.ReLU(),
                        nn.Linear(in_features=5,out_features=1),).to(device)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)

torch.manual_seed(42)

epochs = 6000

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):

    model_0.train()

    # 1. forward pass

    y_logits = model_0(X_train).squeeze()

    y_preds = torch.round(torch.sigmoid(y_logits))

    # 2. Loss function

    loss = loss_fn(y_logits, 
                   y_train)

    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_preds)


    # 3. optimizer zero grad

    optimizer.zero_grad()
    # 4. perform backpropagation

    loss.backward()
    # 5. step the optimizer

    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,y_test)
        test_acc = accuracy_fn(y_true=y_test,
                      y_pred=test_pred)



    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(test_loss)

    if test_loss < 0.1:
        break


np_loss_values = np.array(torch.tensor(loss_values).numpy()) 
np_test_loss_values = np.array(torch.tensor(test_loss_values).numpy()) 


plt.plot(epoch_count,np_loss_values, label='Train Loss')
plt.plot(epoch_count,np_test_loss_values, label='Test Loss')
plt.title('training and test loss curves')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

dirname = os.path.dirname(__file__)
hel_func_path = os.path.join(dirname,'helper_functions.py')


if Path(hel_func_path).is_file():
    print('already downloaded')
else:
    request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
    with open(hel_func_path,'wb') as f:
        print('downloading')
        f.write(request.content)

from helper_functions import plot_predictions,plot_decision_boundary

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('train')
plot_decision_boundary(model_0,X_train,y_train)
plt.subplot(1,2,2)
plt.title('test')
plot_decision_boundary(model_0,X_test,y_test)
plt.show()