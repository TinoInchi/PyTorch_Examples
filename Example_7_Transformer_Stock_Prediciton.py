import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm



class StockTransformer(nn.Module):
    def __init__(self, input_size, d_model, hidden_size, output_size, num_layers, num_heads, dropout):
        super(StockTransformer, self).__init__()
        self.fc_input =nn.Sequential(nn.Linear(input_size,
                                    d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dropout=dropout,
        )
        self.fc = nn.Linear(d_model, hidden_size)
        self.relu = nn.ReLU()
        self.fc_output = nn.Linear(hidden_size, output_size)

    def forward(self, x, y):
        x = self.fc_input(x)
        y = self.fc_input(y)
        x = self.transformer(x,y)
        x = self.fc(x[-1, :, :])  # Take the last hidden state
        x = self.relu(x)
        x = self.fc_output(x)
        return x


# Define a function to create input and target sequences
def create_sequences(data, in_seq_length,out_seq_length):
    input_sequences, target_sequences = [], []
    num_samples = len(data) - in_seq_length - out_seq_length
    input_sequences = torch.tensor([data[i:i+in_seq_length] for i in range(num_samples)]).unsqueeze(-1).float()
    target_sequences = torch.tensor([data[i+in_seq_length:i+in_seq_length+out_seq_length] for i in range(num_samples)]).unsqueeze(-1).float()
    input_sequences=input_sequences.squeeze(-1)
    target_sequences=target_sequences.squeeze(-1)
    return input_sequences, target_sequences



# Hyperparameters and Input

IN_SEQ_LENGTH = 10
OUT_SEQ_LENGTH = 5
RANDOM_SEED = 42
INPUT_SIZE = 1
HIDDEN_SIZE = 64
OUTPUT_SIZE = OUT_SEQ_LENGTH -1
NUM_LAYERS = 8
NUM_HEADS = 4
DROPOUT = 0.1
NUM_EPOCHS = 150
LEARNING_RATE = 0.001
BATCH_SIZE = 16
D_MODEL = NUM_HEADS*16

# Get data set in pandas format
bitcoin_ticker = yf.Ticker("BTC-USD")
BTC_Data = bitcoin_ticker.history(period="max")


del BTC_Data["Dividends"]
del BTC_Data["Stock Splits"]
del BTC_Data["High"]
del BTC_Data["Low"]
del BTC_Data["Open"]
del BTC_Data["Volume"]



BTC_Data = BTC_Data.loc['2022-12-31':].copy()


scaler = MinMaxScaler()
BTC_Data_Scaled = scaler.fit_transform(BTC_Data['Close'].values.reshape(-1, 1))

# Convert data to PyTorch tensors
#prices = torch.FloatTensor(BTC_Data['Close'].values).view(-1, 1, 1)

X, y = create_sequences(BTC_Data_Scaled, IN_SEQ_LENGTH,OUT_SEQ_LENGTH)

'''
# Split data into training and testing sets
train_data, test_data, train_targets, test_targets = train_test_split(  X,
                                                                        y,
                                                                        test_size = 0.0,
                                                                        random_state = RANDOM_SEED,
                                                                        shuffle = False)
'''

test_data = torch.movedim(X,1,0)
test_targets = torch.movedim(y,1,0)

# Create DataLoader
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create Model
model = StockTransformer(INPUT_SIZE,D_MODEL, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, NUM_LAYERS, DROPOUT)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_viewer = []

# Training loop
for epoch in tqdm(range(NUM_EPOCHS)):
    print(f"Epoch: {epoch}\n------")
    for inputs, targets in train_loader:
        inputs = torch.movedim(inputs,1,0)
        targets = torch.movedim(targets,1,0)
        optimizer.zero_grad()
        outputs = model(inputs,targets[:-1,:,:])
        targets = targets.squeeze(-1)
        targets = torch.movedim(targets,1,0)
        loss = criterion(outputs, targets[:,1:])
        loss.backward()
        optimizer.step()
    
    loss_viewer.append(loss.item())


    if epoch % 10 == 9:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')


plt.plot(loss_viewer, label='Loss Behavior')
plt.legend()
plt.show()


# Test the model

src = torch.tensor(BTC_Data_Scaled[-IN_SEQ_LENGTH:]).unsqueeze(1).float()
tgt = torch.zeros(OUTPUT_SIZE, 1, 1)

model.eval()
with torch.inference_mode():
    prediction = model(src, tgt)
    tgt = prediction


# Convert predictions back to original scale
predicted_prices = scaler.inverse_transform(tgt.numpy().reshape(-1, 1))
BTC_Data = BTC_Data.to_numpy()
x_ax_for_prediction = np.arange(len(BTC_Data)-1,len(BTC_Data)-1+OUTPUT_SIZE,1)
# Compare predictions with actual prices
plt.plot(BTC_Data, label='Actual Prices')
plt.plot(x_ax_for_prediction,predicted_prices, label='Predicted Prices')
plt.legend()
plt.show()

