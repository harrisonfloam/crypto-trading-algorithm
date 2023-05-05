## Simple Crypto LSTM Model Class
# Harrison Floam, 25 April 2023

# Import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from cryptomodel import CryptoDataset

class SimpleLSTM(nn.Module):
    """
    A class for creating a simple LSTM neural network model.

    ### Methods:
    -----------
    - __init__(self, input_size, hidden_size, output_size)
        Initializes the LSTM model with the specified input, and output sizes.
    - create_sequences(self, data, seq_length=10)
        Converts the historical price data to sequences and labels for training.
    - train(self, data, batch_size=32, epochs=10)
        Trains the LSTM model on the given historical price data.
    - predict(self, sequence)
        Predicts the next price in a given sequence using the trained LSTM model.

    """

    def __init__(self, input_size, hidden_size, output_size=1, lstm_layers=2, dropout=0.6, verbose=False):
        super().__init__()  # Inherit PyTorch NN Class

        # Define model parameters
        self.criterion = nn.MSELoss()                                # MSE loss function
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers  # Number of LSTM layers
        self.dropout = dropout          # Dropout

        # Define the layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=lstm_layers, dropout=dropout)      # LSTM layer
        self.fc1 = nn.Linear(hidden_size + input_size, output_size)                   # Fully-connected layer 1
        self.activation = nn.Sigmoid()  # Sigmoid Activation Function

        # Define other parameters
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)     # Adam optimizer
        self.verbose = verbose  # Verbose debug flag

    # Define the forward function
    def forward(self, x):
        batch_size = x.shape[0]
        
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, requires_grad=True)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, requires_grad=True)

        _, (hn, _) = self.lstm(x, (h0, c0))     # Pass through LSTM layer
        out = self.fc1(torch.cat([hn[-1], x[:, -1]], dim=1))    # Pass final hidden tensor from all LSTM layers through fully-connected layer
        out = self.activation(out)
        
        return out
    
    # Train the model
    def train(self, data, seq_length, batch_size=32, epochs=10):

        # Create DataLoader
        dataset = CryptoDataset(data=data, seq_length=seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Train the model
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data_load in enumerate(dataloader):
                inputs, labels = data_load
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                print(f'Loss {loss}')
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            if self.verbose: print(f'Epoch {epoch+1} loss: {running_loss/len(dataloader):.6f}') # Print running loss if verbose

    # Query the model
    def predict(self, data, hidden):
        self.eval()     # Toggle evaluation mode
        with torch.no_grad():
            input_seq = data.iloc[-1:, :].values                        # Last row of dataframe (most recent features)
            input_seq = torch.tensor(input_seq).unsqueeze(1).float()
            output, _ = self(input_seq, hidden)
            predicted_price = output.item()
            confidence = 1.0 - self.criterion(output, input_seq[:, -1:, :]).item()
        return predicted_price, confidence
    
    # Update the model with new data ("online training")
    def update_model(self, data, seq_length=1):
        input_seq, target_seq = self.create_sequences(data=data, seq_length=seq_length)
        self.optimizer.zero_grad()  # Clear the gradients from the optimizer

        output, self.hidden = self(input_seq, self.hidden)  # Pass the input sequence and previous hidden state through the model
        loss = self.criterion(target_seq, output)  # Compute the loss between the predicted and actual values

        loss.backward()  # Backpropagate loss
        self.optimizer.step()  # Update model parameters
