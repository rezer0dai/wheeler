import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.policy import *

# following paper : https://arxiv.org/abs/1509.02971
# code sourced from UDACITY -> https://github.com/udacity/deep-reinforcement-learning
# minor updates to adapt to my framewok + striped bias at final layers !

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, wrap_action, cfg, fc1_units=400, fc2_units=300):
        super().__init__()
        state_size = state_size + cfg['her_state_features']
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias = False)
        self.fc3 = nn.Linear(fc2_units, action_size, bias = False)
        self.reset_parameters()

        self.algo = DDPG(wrap_action) if cfg['ddpg'] else PPO(action_size)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, goal, state):
        if goal.dim() == state.dim():
            state = torch.cat([goal, state], -1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.algo(x)

    def sample_noise(self):
        return
    def remove_noise(self):
        return

class Critic(nn.Module):
    def __init__(self, state_size, action_size, wrap_value, cfg, fcs1_units=400, fc2_units=300):
        super().__init__()
        self.wrap_value = wrap_value
        state_size = state_size + cfg['her_state_features']
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units, bias = False)
        self.fc3 = nn.Linear(fc2_units, 1, bias = False)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)

    def forward(self, goal, state, action):
        if goal.dim() == state.dim():
            state = torch.cat([goal, state], -1)
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.wrap_value(x)
