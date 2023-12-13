from sqlite3 import Time
import yfinance as yf
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_stacked_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_stacked_layers = num_stacked_layers
        
        self.lstm = nn.LSTM(input_dim, 
                            hidden_dim, 
                            num_stacked_layers, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, 
                            output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, 
                         batch_size, 
                         self.hidden_dim).requires_grad_()

        c0 = torch.zeros(self.num_stacked_layers, 
                         batch_size, 
                         self.hidden_dim).requires_grad_()

        out, _ = self.lstm(x, 
                           (h0, c0))

        out = self.fc(out[:, -1, :]) 

        return out

class TimeSeriesDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,i):
        return self.X[i],self.y[i]

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

def prepare_dataframe_lstm(df,n_steps):
    
    for i in range(1, n_steps+1,1):
        df[f'Close(t-{i})'] = df ['Close'].shift(i)

    df.dropna(inplace=True)

    return df


def train_one_epoch():
    model.train(True)
    for batch_index,batch in enumerate(train_loader):
        
        x_batch,y_batch = batch[0],batch[1]
        output = model(x_batch)
        loss = loss_fn(output,
                        y_batch)

        acc = accuracy_fn(y_true=y_batch,
                          y_pred=output)

        # 3. optimizer zero grad

        optimizer.zero_grad()

        # 4. perform backpropagation

        loss.backward()

        # 5. step the optimizer

        optimizer.step()

        loss_values.append(loss)

        if batch_index % 100 == 99:
            print(f'Loss: {loss}')

    epoch_count.append(epoch)

def test_one_epoch():
    model.eval()
    for batch_index,batch in enumerate(train_loader):
        x_batch,y_batch = batch[0],batch[1]
        with torch.inference_mode():

            output = model(x_batch)

            test_loss = loss_fn(output,y_batch)
            test_acc = accuracy_fn(y_true=y_batch,
                          y_pred=output)

            test_loss_values.append(test_loss)
            
    print(f'batch number: {batch_index}')
    print(f'Test Loss: {test_loss}')

        
        

bitcoin_ticker = yf.Ticker("BTC-USD")
BTC_Data = bitcoin_ticker.history(period="max")


del BTC_Data["Dividends"]
del BTC_Data["Stock Splits"]
del BTC_Data["High"]
del BTC_Data["Low"]
del BTC_Data["Open"]
del BTC_Data["Volume"]



BTC_Data = BTC_Data.loc['2022-12-31':].copy()


# Split in Training data and Test data
lookback = 30
BTC_Data = prepare_dataframe_lstm(BTC_Data,lookback)
BTC_Data_np = BTC_Data.to_numpy()

scaler = MinMaxScaler(feature_range=(-1,1))
scaled_BTC_Data = scaler.fit_transform(BTC_Data)

X = scaled_BTC_Data[:,1:]
X = np.copy(np.flip(X,axis=1))
y = scaled_BTC_Data[:,0]


X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split( X,
                                                     y,
                                                     test_size= 0.1,
                                                     random_state = 42,
                                                     shuffle=False)

X_train = X_train.reshape((-1,lookback,1))
X_test = X_test.reshape((-1,lookback,1))
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)

train_dataset = TimeSeriesDataset(X_train,y_train)
test_dataset = TimeSeriesDataset(X_test,y_test)

batch_size = 16
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)



                                         
# Initialize model

input_dim = 1
hidden_dim = 20
num_layers = 2
output_dim = 1
learning_rate = 0.01
epochs = 40

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_stacked_layers=num_layers)



# Setting Loss Function 

loss_fn = nn.MSELoss(reduction='mean')

# Set Optimizer

optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)

# set random seed

torch.manual_seed(42)

# initialize arrays to see progress

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    # training loop
    train_one_epoch()
    # testing loop
    test_one_epoch()




with torch.inference_mode():
    predicted = model(X_train).detach().numpy()


train_prediction = predicted.flatten()

dummies = np.zeros((X_train.shape[0],lookback+1))
dummies[:,0] = train_prediction
dummies = scaler.inverse_transform(dummies)

train_predictions = np.copy(dummies[:,0])
#train_predictions = np.expand_dims(train_predictions,axis=1)

dummies = np.zeros((X_train.shape[0],lookback+1))
dummies[:,0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = np.copy(dummies[:,0])
#new_y_train = np.expand_dims(new_y_train,axis=1)


test_prediction = model(X_test).detach().numpy().flatten()

dummies = np.zeros((X_test.shape[0],lookback+1))
dummies[:,0] = test_prediction
dummies = scaler.inverse_transform(dummies)

test_prediction = np.copy(dummies[:,0])
#test_prediction = np.expand_dims(test_prediction,axis=1)

dummies = np.zeros((X_test.shape[0],lookback+1))
dummies[:,0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = np.copy(dummies[:,0])
#new_y_test = np.expand_dims(new_y_test,axis=1)

print(train_prediction.shape,test_prediction.shape)
print(new_y_train.shape,new_y_test.shape)


tot_prediction = np.append(train_predictions,test_prediction)
tot_actual = np.append(new_y_train,new_y_test)



plt.plot(tot_prediction[1:], label='Actual Close')
plt.plot(tot_actual,label='Predicted Close')
plt.xlabel('Day')
plt.legend()
plt.show()
