import torch
import torch.nn as nn
import torch.nn.init as init
from utils import activation_fun


class FCNN(nn.Module):
    """
    Fully Connected Neural Network (FCNN) configurable with Optuna parameters.

    Supports:
    - Flexible number of layers
    - BatchNorm or LayerNorm or no normalization
    - Dropout / AlphaDropout
    - Flexible activation functions
    - He/Xavier initialization based on activation

    Args:
        input_dim (int): Input feature size (number of genes)
        label_dim (int): Number of output classes 
        params (dict): Hyperparameters dictionary
    """
    
    def __init__(self, input_dim, label_dim, params):
        super().__init__()

        self.input_dim = input_dim
        self.dropout_rate = params['dropout']
        self.num_layers = params['num_layers']
        self.fc_dims = [params[f'dim_{i}'] for i in range(1, self.num_layers + 1)]

        # Activation function
        self.activation = activation_fun(act=params["activation"])

        # Dropout selection
        if params['dropout_type'] == 'simple_drop':
            self.dropout = nn.Dropout(p=self.dropout_rate, inplace=False)
        else:
            self.dropout = nn.AlphaDropout(p=self.dropout_rate, inplace=False)

        # Normalization type
        self.normalization = params['normalization']

        # Create sequential encoder layers
        self.encoder = self._build_encoder_layers()

        # Classifier (output layer)
        self.classifier = nn.Linear(self.fc_dims[-1], label_dim)

    def _build_encoder_layers(self):
        """
        Build the encoder (hidden layers) of the FCNN.
        Returns:
            nn.Sequential: Encoder model
        """
        
        layers = nn.ModuleList()

        for i in range(self.num_layers):
            # Define input/output dimensions for ith layer
            in_dim = self.input_dim if i == 0 else self.fc_dims[i - 1]
            out_dim = self.fc_dims[i]

            # Linear layer
            fc_layer = nn.Linear(in_dim, out_dim)

            # Weight initialization based on activation function
            self._initialize_weights(fc_layer)

            # Build layer sequentially
            layer = nn.Sequential(fc_layer)

            # Add normalization
            if self.normalization == 'batch':
                layer.add_module('norm', nn.BatchNorm1d(out_dim))
            elif self.normalization == 'layer':
                layer.add_module('norm', nn.LayerNorm(out_dim))

            # Add activation and dropout
            layer.add_module('activation', self.activation)
            layer.add_module('dropout', self.dropout)

            layers.append(layer)

        return nn.Sequential(*layers)

    def _initialize_weights(self, layer):
        """
        Apply appropriate weight initialization based on activation function.
        Args:
            layer (nn.Linear): Linear layer to initialize
        """
        
        if isinstance(self.activation, nn.ReLU):
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        elif isinstance(self.activation, nn.LeakyReLU):
            init.kaiming_uniform_(layer.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        elif isinstance(self.activation, nn.ELU):
            init.kaiming_uniform_(layer.weight, a=1.0, mode='fan_in', nonlinearity='leaky_relu')
        elif isinstance(self.activation, nn.Tanh):
            init.xavier_uniform_(layer.weight, gain=5 / 3)
        else:
            init.xavier_uniform_(layer.weight, gain=1)

        init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Tuple (input features, output logits)
        """
        input = x
        encoded = self.encoder(x)
        logits = self.classifier(encoded)

        return input, logits


