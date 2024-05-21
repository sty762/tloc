from matplotlib.pylab import RandomState
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn.ensemble import RandomForestRegressor

class Encoder_conv(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder_conv, self).__init__()
        ### Convolutional section
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True))
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(288, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, encoded_space_dim))

    def forward(self, x):
        #x=x.type(torch.DoubleTensor)
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Encoder_conv_with_prior(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder_conv_with_prior, self).__init__()
        ### Convolutional section
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True))
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.prior_fc = nn.Sequential(
            nn.Linear(encoded_space_dim, 8),
            #nn.ReLU(True),
            nn.Dropout(0.5))
        self.encoder_lin = nn.Sequential(
            nn.Linear(296, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, encoded_space_dim))

    def forward(self, x, prior):
        #x=x.type(torch.DoubleTensor)
        if torch.isnan(x).any():
            itay=29
        x = self.encoder_conv(x)
        x = self.flatten(x)
        prior = self.prior_fc(prior)
        comb = torch.cat((prior, x), 1)
        out = self.encoder_lin(comb)
        return out
    
class Encoder_tloc(RandomForestRegressor):
    def __init__(self, **kwargs):
        super(Encoder_tloc, self).__init__(kwargs)

    def __call__(self, **kwargs) -> torch.Any:
        return self.predict(kwargs)