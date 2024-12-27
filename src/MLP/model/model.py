import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, params, input_size):
        """
        Initialize MLP with dynamic architecture based on parsed parameters
        
        Args:
            params (dict): Contains:
                - num_layers (int): Number of hidden layers
                - layer_sizes (list): Size of each hidden layer
                - dropouts (list): Dropout rate for each hidden layer
                - activations (list): Activation function for each hidden layer
            input_size (int): Size of input features
        """
        super(MLP, self).__init__()
        
        # Store architecture parameters
        self.num_layers = params['num_layers']
        self.layer_sizes = [input_size] + params['layer_sizes']
        self.dropouts = params['dropouts']
        self.activations = params['activations']
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        
        # Build all layers
        for i in range(self.num_layers):
            # Add linear layer
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
            
            # Add activation
            if self.activations[i] == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activations[i] == 'leaky_relu':
                self.layers.append(nn.LeakyReLU())
            
            # Add dropout only for the last layer (layer3)
            if i == self.num_layers - 1 and self.dropouts[i] > 0:
                self.layers.append(nn.Dropout(self.dropouts[i]))
        
        # Final output layer
        self.output_layer = nn.Linear(self.layer_sizes[-1], 1)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Pass through all hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Pass through output layer
        x = self.output_layer(x)
        return x
    
    def get_architecture_info(self):
        """Return a string describing the network architecture"""
        architecture = []
        architecture.append(f"Input size: {self.layer_sizes[0]}")
        
        for i in range(self.num_layers):
            dropout_info = f", dropout={self.dropouts[i]:.2f}" if i == self.num_layers - 1 else ""
            layer_info = (f"Hidden layer {i+1}: "
                         f"size={self.layer_sizes[i+1]}, "
                         f"activation={self.activations[i]}"
                         f"{dropout_info}")
            architecture.append(layer_info)
            
        architecture.append(f"Output size: 1")
        return "\n".join(architecture)