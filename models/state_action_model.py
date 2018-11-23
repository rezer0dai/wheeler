import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
    nn.init.kaiming_uniform_(layer.weight)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, wrap_value, cfg, fcs1_units=400, fc2_units=300):
        super().__init__()
        self.wrap_value = wrap_value

        state_size += cfg['her_state_features']
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fca1 = nn.Linear(action_size, fc2_units)
        self.fc2 = nn.Linear(fcs1_units + fc2_units, fc2_units, bias = False)
        self.fc3 = nn.Linear(fc2_units, 1, bias = False)

        self.apply(initialize_weights)

    def forward(self, goal, state, action):
        if goal.dim() == state.dim():
            state = torch.cat([goal, state], -1)
        xs = F.relu(self.fcs1(state))
        xa = F.relu(self.fca1(action))
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.wrap_value(x)
