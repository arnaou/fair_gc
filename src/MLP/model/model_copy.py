import torch
import torch.nn as nn
import nni

class MLP(nn.Module):
    def __init__(self, params, input_size=None):
        """
        Initialize MLP with flexible input size
        
        Args:
            params (dict): Model hyperparameters
            input_size (int, optional): Input feature dimension. 
                                      If None, will default to 424
        """
        super(MLP, self).__init__()
        # Set input size with fallback to 424 if not provided
        self.input_size = input_size if input_size is not None else 424
        
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size, params['hidden_size_1']),
            nn.ReLU() if params['activation_1'] == 'relu' else nn.LeakyReLU(),
            nn.Dropout(params['dropout_1'])
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(params['hidden_size_1'], params['hidden_size_2']),
            nn.ReLU() if params['activation_2'] == 'relu' else nn.LeakyReLU(),
            nn.Dropout(params['dropout_2'])
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(params['hidden_size_2'], params['hidden_size_3']),
            nn.ReLU() if params['activation_3'] == 'relu' else nn.LeakyReLU()
        )
        
        self.output = nn.Linear(params['hidden_size_3'], 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x