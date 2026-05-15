import yaml
import gymnasium as gym 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, output_activation=None):
        super(MLP, self).__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if hidden_dims is None:
            hidden_dims = []
        
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])
        self.output_activation = output_activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        if self.output_activation == 'softmax':
            x = F.softmax(x, dim=1)
        return x

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(PolicyNet, self).__init__()
        self.model = MLP(input_dim=state_dim, hidden_dims=hidden_dims, output_dim=action_dim, output_activation='softmax')

    def forward(self, x):
        return self.model(x)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super(ValueNet, self).__init__()
        self.model = MLP(input_dim=state_dim, hidden_dims=hidden_dims, output_dim=1, output_activation=None)

    def forward(self, x):
        return self.model(x)