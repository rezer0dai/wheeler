import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

import numpy as np

from utils.nes import *
from utils.policy import *

def initialize_weights(layer):
    if not isinstance(layer, nn.Linear):
        return
    nn.init.xavier_uniform_(layer.weight)

class ActorNN(nn.Module):
    def __init__(self, task, cfg):
        super(ActorNN, self).__init__()
        self.algo = DDPG(task) if cfg['ddpg'] else PPO(task)
        self.action_size = task.action_size()
        self.cfg = cfg

        self.rnn = nn.LSTMCell(
                task.state_size(), cfg['history_features'] // 2)

        if self.cfg['her_state_size']:
            self.her_state = nn.Linear(cfg['her_state_size'], cfg['her_state_features'])

        self.apply(initialize_weights)

        self.state_size = task.state_size()

        self.ex = nn.Linear(cfg['her_state_features'] + cfg['history_features'] // 2,
                task.action_size())

        self.concat = NoisyLinear(cfg['her_state_features'] + cfg['history_features'] // 2, 64)
        self.output =  NoisyLinear(64, task.action_size())

        self.features = None

    def forward(self, state, hidden):
        x = state[:, self.cfg['her_state_size']:]

        x = x.view(x.size(0), self.cfg['history_count'], -1)

        out, feature = self.rnn(x.transpose(0, 1)[0, :], hidden.view(2, state.size(0), -1))
        self.features = torch.cat([out, feature], 1).view(1, 1, -1)

        for s in x.transpose(0, 1)[1:, :]:
            out, feature = self.rnn(s, (out, feature))

        x = out
        if self.cfg['her_state_size']:
            her_state = F.tanh(self.her_state(
                            state[:, :self.cfg['her_state_size']]))
            x = torch.cat([x, her_state], dim=1)

        return self.algo(self.ex(x))

        x = F.relu(self.concat(x))
        x = self.output(x)
        return self.wrap_action(x)

    def sample_noise(self):
        self.ex.sample_noise()
        return
        self.concat.sample_noise()
        self.output.sample_noise()

    def remove_noise(self):
        pass

    def recomputable(self):
        return True#False#

    def extract_features(self, states):
        states = states[:, self.cfg['her_state_size']:self.cfg['her_state_size']+self.state_size:]
        states = torch.from_numpy(states)

        hidden = torch.zeros(
                    2, 1, self.cfg['history_features'] // 2)
        feature = hidden.view(1, 1, -1)
        features = [feature.detach().cpu().numpy()]

        for x in states:
            x = x.view(1, -1)
            out, feature = self.rnn(x, hidden)
            hidden = (out, feature)
            feature = torch.cat(hidden, 1).view(1, 1, -1)
            features.append(feature.detach().cpu().numpy())

        return features[:-1]
