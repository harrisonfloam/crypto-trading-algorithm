## Crypto Model Class
# Harrison Floam, 18 April 2023

# Import
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CryptoModel:
    """A wrapper class for different cryptocurrency trading models.
    
    ### Parameters:
    -----------
    - model_class_name: string
        The trading model class to be used (e.g. CryptoLSTM).
    - **kwargs:
        Additional arguments to be passed to the model class constructor.
    """
    def __init__(self, model_class_name, **kwargs):
        model_module = __import__('models.' + model_class_name.lower(), fromlist=[model_class_name])
        model_class = getattr(model_module, model_class_name)
        self.model = model_class(**kwargs)

    def train(self, *args, **kwargs):
        self.model.train(*args, **kwargs)

    def predict(self, *args, **kwargs):
        self.model.predict(*args, **kwargs)

    def update_model(self, *args, **kwargs):
        self.model.update_model(*args, **kwargs)
        

class CryptoDataset(Dataset):
    """
    A class for creating a PyTorch dataset from historical price data.
    """
    def __init__(self, data, seq_length):
        self.sequences, self.labels = self.create_sequences(data, seq_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.sequences[index]).float(), torch.tensor(self.labels[index]).float()

    def create_sequences(self, data, seq_length):
        """
        Convert historical price data to sequences and labels for training using overlapping windows.
        """
        num_sequences = len(data) - seq_length
        sequences = []
        labels = []

        for i in range(num_sequences):
            sequence = data.iloc[i:i+seq_length, :].values
            label = data.loc[i + seq_length, 'close']
            sequences.append(sequence)
            labels.append(label)

        return sequences, labels
