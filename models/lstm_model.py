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
    def __init__(self, state_size, action_size, wrap_action, cfg):
        super(ActorNN, self).__init__()
        self.algo = DDPG(wrap_action) if cfg['ddpg'] else PPO(action_size)
        self.action_size = action_size
        self.cfg = cfg

        self.rnn = nn.LSTMCell(
                state_size, cfg['history_features'] // 2)

        if self.cfg['her_state_size']:
            self.her_state = nn.Linear(cfg['her_state_size'], cfg['her_state_features'])

        self.state_size = state_size

        self.ex = NoisyLinear(
                cfg['her_state_features'] + cfg['history_features'] // 2,
                action_size)

#        self.ex = nn.Linear(
#                cfg['her_state_features'] + cfg['history_features'] // 2,
#                action_size)

        self.concat = NoisyLinear(cfg['her_state_features'] + cfg['history_features'] // 2, 64)
        self.output =  NoisyLinear(64, action_size)

        self.apply(initialize_weights)

        self.features = None
        self.bn = nn.BatchNorm1d(state_size)

    def forward(self, state, hidden):
        x = state[:, self.cfg['her_state_size']:]

        x = x.view(x.size(0), self.cfg['history_count'], -1)

        if self.cfg['use_batch_norm']:
            x = self.bn(x.transpose(1, 2)).transpose(1, 2)

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

    def extract_features(self, states):
        states = states[:, 
                self.cfg['her_state_size']:self.cfg['her_state_size']+self.state_size:]
        states = torch.from_numpy(states)

        if self.cfg['use_batch_norm']:
            states = states.unsqueeze(0).transpose(1, 2)
            states = self.bn(states).transpose(1, 2).squeeze(0)

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
