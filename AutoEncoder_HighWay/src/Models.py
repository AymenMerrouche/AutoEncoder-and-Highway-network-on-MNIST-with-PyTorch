import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import pdb
import torch.optim as optim


# Dataset to encapsulate Minst dataset
class MonDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __getitem__(self, index):
        # flatten and normalize
        image = torch.tensor(self.data[index].reshape(-1)/255, dtype=torch.float32)
        return (image, torch.tensor(self.labels[index].reshape(-1)))
    def __len__(self):
        return self.data.shape[0]

# state class for checkpointing
class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0

# AutoEncoder with tied weights module
class AutoEncoder(nn.Module):
    def __init__(self, inputShape, projectionShape):
        super(AutoEncoder, self).__init__()
        self.inputShape = inputShape # dimension of initial space
        self.projectionShape = projectionShape # dimension of the projection space (reduced dimension)

        # Define and initialize the parameters of the Linear layers
        # Tied autoencoder : the two linear layers share the same weights
        # We explicitly define bias and weights to tie the two linear layers of the autoencoder
        # We will use nn.functional for the linear layers
        # y = x@A.T + b
        self.w = torch.nn.Parameter(torch.zeros((self.projectionShape, self.inputShape), dtype=torch.float32))
        self.b_encding = torch.nn.Parameter(torch.rand(projectionShape, dtype=torch.float32))
        self.b_decoding = torch.nn.Parameter(torch.rand(inputShape, dtype=torch.float32))

        # activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def encode(self, x):
        # encoding : Linear -> ReLU
        return self.relu(F.linear(x.float(), self.w, self.b_encding))
    def decode(self, y):
        # decoding using transpose(w) : Linear -> Sigmoid
        return self.sigmoid(F.linear(y.float(),self.w.t(), self.b_decoding))
    def forward(self, x):
        # Linear -> ReLU -> Linear -> Sigmoid
        return self.decode(self.encode(x))


# one HighWay Layer
class OneHighWayLayer(nn.Module):
    def __init__(self, input_size, hidden_size, f, bias_T):
        super(OneHighWayLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.f = f #either ReLU or tanh
        self.bias_T = bias_T

        self.H = nn.Linear(self.hidden_size, self.hidden_size)
        self.T = nn.Linear(self.hidden_size, self.hidden_size)
        #  the network is initiallybiased towardscarrybehavior
        self.T.bias.data.fill_(self.bias_T)
    def forward(self, x):
        # Just apply the formula y=H(x,WH)·T(x,WT) +x·(1−T(x,WT))
        t = F.sigmoid(self.T(x))
        h = self.f(self.H(x))
        out = h*t + x*(1 - t)
        return out

# HighWay network (H*T + x*(1-T))
class HighWay(nn.Module):
    def __init__(self, input_size, hidden_size, nb_layers, nb_classes, f = nn.ReLU(), bias_T = -3):
        super(HighWay, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.f = f #either ReLU or tanh
        self.nb_layers = nb_layers
        self.nb_classes = nb_classes
        self.bias_T = bias_T

        # 1st layer is fully connected
        self.first_layer = nn.Linear(input_size, self.hidden_size)
        # last layer to classify
        self.last_layer = nn.Linear(self.hidden_size, self.nb_classes)

        # highway layers as defined above
        self.HighWayLayers = nn.ModuleList([OneHighWayLayer(self.input_size, self.hidden_size, self.f, self.bias_T)\
                                for i in range(nb_layers)])


    def forward(self, x):
        # first fully connected layer
        out = self.first_layer(x)

        # unroll the highway layers
        for high_way_layer in self.HighWayLayers:
            out = high_way_layer(out)
        # classify the result
        return self.last_layer(out)
