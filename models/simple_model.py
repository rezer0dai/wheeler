import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nes import *

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
    print("OK", layer)
#    nn.init.kaiming_uniform_(layer.weight)
    nn.init.xavier_uniform_(layer.weight)

class CriticNes(nn.Module):
    def __init__(self, task, cfg):
        super(CriticNes, self).__init__()
        self.wrap_value = task.wrap_value

        self.cfg = cfg

        self.state_dim = cfg['her_state_size'] + task.state_size() * cfg['history_count']
        self.action_dim = task.action_size()

        self.net = NoisyNet([self.state_dim + self.action_dim, 256, 256,  256, 1])

        self.apply(initialize_weights)

    def forward(self, state, action, _):
        x = torch.cat([state, action], 1)
        return self.wrap_value(self.net(x))

    def parameters(self):
        return self.net.parameters()

    def sample_noise(self):
        self.net.sample_noise()

    def remove_noise(self):
        return

class CriticNN(nn.Module):

    def __init__(self, task, cfg):
        super(CriticNN, self).__init__()
        self.wrap_value = task.wrap_value

        self.cfg = cfg

        self.state_dim = cfg['her_state_size'] + task.state_size() * cfg['history_count']
        self.action_dim = task.action_size()

        self.net = nn.Sequential(
                nn.Linear(self.state_dim + self.action_dim,256, bias=False),
                nn.ReLU(),
                nn.Linear(256,256, bias=False),
                nn.ReLU(),
                nn.Linear(256,256, bias=False),
                nn.ReLU(),
                nn.Linear(256,1, bias=False),
                )

        self.apply(initialize_weights)

    def forward(self, state, action, _):
        x = torch.cat([state, action], 1)
        return self.wrap_value(self.net(x))

    def sample_noise(self):
        return
    def remove_noise(self):
        return

class ActorNN(nn.Module):

    def __init__(self, task, cfg):
        super(ActorNN, self).__init__()

        self.cfg = cfg

        self.state_dim = cfg['her_state_size'] + task.state_size() * cfg['history_count']
        self.action_dim = task.action_size()
        self.wrap_action = task.wrap_action

        self.net = nn.Sequential(
                nn.Linear(self.state_dim,256, bias=False),
                nn.ReLU(),
                nn.Linear(256,256, bias=False),
                nn.ReLU(),
                nn.Linear(256,256, bias=False),
                nn.ReLU(),
                nn.Linear(256,self.action_dim, bias=False),
                )

        self.apply(initialize_weights)

    def forward(self, state, _):
        self.features = torch.zeros(state.size(0), 1, self.cfg['history_features'])
        return self.wrap_action(self.net(state))

    def sample_noise(self):
        return

    def remove_noise(self):
        return
    def recomputable(self):
        return False
