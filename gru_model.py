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

        self.her_state = nn.Linear(cfg['her_state_size'], cfg['her_state_features'])
        self.rnn = nn.GRU(task.state_size(), cfg['action_features'], batch_first=True)
        self.state_action = nn.Linear(cfg['her_state_features'] + cfg['action_features'] * cfg['history_count'] + self.action_size, 300)
        self.value = nn.Linear(300, 1)
        self.apply(initialize_weights)

    def forward(self, state, action, history):
        her_state = state[:, :self.cfg['her_state_size']]
        her_state = self.her_state(her_state)

        state = state[:, self.cfg['her_state_size']:]
        x = state.view(state.size(0), self.cfg['history_count'], -1)
        x, _ = self.rnn(x, history)

        x = x.contiguous().view(x.size(0), -1)
        x = torch.cat([x, her_state], dim=1)

#        x = F.dropout(x, .3)
        action = action.view(action.size(0), -1)
        x = F.relu(self.state_action(torch.cat([x, action], 1)))
        x = self.value(x)
        return self.wrap_value(x)
# noisy nets
class ActorNN(nn.Module):
    def __init__(self, task, cfg):
        super(ActorNN, self).__init__()
        self.wrap_action = task.wrap_action
        self.action_size = task.action_size()
        self.cfg = cfg

        self.her_state = nn.Linear(cfg['her_state_size'], cfg['her_state_features'])
        self.rnn = nn.GRU(task.state_size(), cfg['action_features'], batch_first=True)
        self.state_action = NoisyLinear(cfg['her_state_features'] + cfg['action_features'] * cfg['history_count'], 400)
        self.noisy_linear = NoisyLinear(400, self.action_size)
        self.apply(initialize_weights)
        self.features = None

    def forward(self, state, history):
        her_state = state[:, :self.cfg['her_state_size']]
        her_state = self.her_state(her_state)

        state = state[:, self.cfg['her_state_size']:]
        x = state.view(state.size(0), self.cfg['history_count'], -1)
        x, _ = self.rnn(x, history)
        self.features = x.permute(1, 0, 2)[-1].detach().cpu().numpy()

        x = x.contiguous().view(state.size(0), -1)
        x = torch.cat([x, her_state], dim=1)

        #x = F.dropout(x, .2)
        x = F.relu(self.state_action(x))
        x = self.noisy_linear(x)
        return self.wrap_action(x)

    def sample_noise(self):
        self.state_action.sample_noise()
        self.noisy_linear.sample_noise()
    def remove_noise(self):
        self.state_action.remove_noise()
        self.noisy_linear.remove_noise()
