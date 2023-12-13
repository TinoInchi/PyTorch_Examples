from cgi import test
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import os
from pathlib import Path


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


def plot_prediction(train_data,train_labels,test_data,test_labels,predictions):

    fig_data = plt.figure(2,figsize=(10,7))

    plt.scatter(train_data, train_labels, c='b', s = 4, label='Training data')

    plt.scatter(test_data, test_labels, c='g', s = 4, label='Testing data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s = 4, label='Predictions')
    
    plt.legend(prop={"size":14})
    plt.show()

    
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split],y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

model_0 = LinearRegression()

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.0001)

epochs = 30
epoch = 1
torch.manual_seed(42)

epoch_count = []
loss_values = []
test_loss_values = []

#for epoch in range(epochs):

while (True):

    model_0.train()

    # 1. forward pass

    y_preds = model_0(X_train)
    # 2. Loss function

    loss = loss_fn(y_preds, y_train)
    # 3. optimizer zero grad

    optimizer.zero_grad()
    # 4. perform backpropagation

    loss.backward()
    # 5. step the optimizer

    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)

        test_loss = loss_fn(test_pred,y_test)

    if loss  < 0.01:
        print(f'Loop Count: {epoch}')
        print(f'Loss: {loss }\n Loss_test: {test_loss }')
        param_list = model_0.state_dict()
        print(param_list)
        break;

    if (epoch % 1000 == 0):
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)

    epoch += 1

np_loss_values = np.array(torch.tensor(loss_values).numpy()) 
np_test_loss_values = np.array(torch.tensor(test_loss_values).numpy()) 

loss_fig = plt.figure(1,figsize=(10,7))

plt.plot(epoch_count,np_loss_values, label='Train Loss')
plt.plot(epoch_count,np_test_loss_values, label='Test Loss')
plt.title('training and test loss curves')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plot_prediction(X_train,y_train,X_test,y_test,test_pred)


dirname = os.path.dirname(__file__)

torch_file_name = os.path.join(dirname,'torch_file_model_0.pth')

torch.save(model_0.state_dict(),torch_file_name)