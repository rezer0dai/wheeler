import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from nes import NoisyLinear

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
    nn.init.kaiming_uniform_(layer.weight)

class CriticNN(nn.Module):
    def __init__(self, task, cfg):
        super(CriticNN, self).__init__()

        self.wrap_value = task.wrap_value
        self.action_size = task.action_size()
        self.cfg = cfg

        self.state_encoder = nn.Linear(
                task.state_size() * cfg['history_count'],
                400)

        self.state_action = nn.Linear(
                400 + self.action_size,
                100)

        self.value = nn.Linear(100, 1)

        self.apply(initialize_weights)

    def forward(self, state, action, _):
        x = F.relu(self.state_encoder(state))
        x = torch.cat([x, action], 1)
        x = F.relu(self.state_action(x))
        x = self.value(x)
        return self.wrap_value(x)

class ActorNN(nn.Module):
    def __init__(self, task, cfg):
        super(ActorNN, self).__init__()
        self.wrap_action = task.wrap_action
        self.action_size = task.action_size()

        self.encoder = nn.Linear(task.state_size() * cfg['history_count'], 400)

        self.noisy_linear = nn.Linear(400, 400)

        self.apply(initialize_weights)

        self.value = NoisyLinear(400, self.action_size)

        self.features = np.zeros(cfg['action_features'])#torch.autograd.Variable(torch.from_numpy(np.zeros(1)))

    def forward(self, state, _):
        x = F.relu(self.encoder(state))
        x = F.relu(self.noisy_linear(x))

        # those are our actions!
        x = self.value(x)
        pi = self.wrap_action(x)

        return pi

    def sample_noise(self):
        self.value.sample_noise()
    def remove_noise(self):
        self.value.remove_noise()
