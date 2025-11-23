import torch
from torch import nn

def make_mlp(input_dim=784, hidden_layers=1, width=128, output_dim=10):
    layers = []
    dim = input_dim
    
    for _ in range(hidden_layers):
        layers.append(nn.Linear(dim, width))
        layers.append(nn.ReLU())
        dim = width

    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)
