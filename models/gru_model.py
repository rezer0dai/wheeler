import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.nes import *

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
    nn.init.xavier_uniform_(layer.weight)

class CriticNN(nn.Module):
    def __init__(self, task, cfg):
        super().__init__()

        self.wrap_value = task.wrap_value
        self.cfg = cfg

        total_size = cfg['history_count'] * task.state_size() + cfg['her_state_size'] + task.action_size()

        self.state_encode = nn.Linear(
                task.state_size() * cfg['history_count'] + task.action_size()
                , 128)

        if cfg['her_state_size']:
            self.goal_encode = nn.Linear(cfg['her_state_size'], cfg['her_state_features'])

        self.q_approx = nn.Sequential(
                nn.Linear(128 + cfg['her_state_features'] + cfg['history_features'], 256),
                nn.ReLU(),
                nn.Linear(256,128, bias=False),
                nn.ReLU(),
                nn.Linear(128, 1, bias=False)
                )

        self.apply(initialize_weights)

    def forward(self, state, action, context):
        if self.cfg['her_state_size']:
            goal = state[:, :self.cfg['her_state_size']]

        context = context.squeeze(0)
        state = F.tanh(self.state_encode(
            torch.cat([action,
            state[:, self.cfg['her_state_size']:]
            ], dim = 1)))

        state = torch.cat([state, context], dim=1)

        if self.cfg['her_state_size']:
            goal = F.tanh(self.goal_encode(goal))
            state = torch.cat([state, goal], dim=1)

        x = self.q_approx(state)
        return self.wrap_value(x)

    def sample_noise(self):
        return
    def remove_noise(self):
        return

# noisy nets
class ActorNN(nn.Module):
    def __init__(self, task, cfg):
        super().__init__()
        self.wrap_action = task.wrap_action
        self.cfg = cfg

# handle Normalization vs obsolete feature vector in replay buffer
        self.state_size = task.state_size()

        self.rnn = nn.GRU(
                self.state_size, cfg['history_features'],
                batch_first=True,
                bidirectional=False,
#                dropout=.2,
                num_layers=1)
        self.features = None

        if self.cfg['her_state_size']:
            self.her_state = nn.Linear(cfg['her_state_size'], cfg['her_state_features'])

        total_size = cfg['her_state_features'] + cfg['history_features']

        self.ex = NoisyLinear(cfg['her_state_features'] + cfg['history_features'], task.action_size())

        self.features = None

        self.apply(initialize_weights)


    def forward(self, state, history):
        x = state[:, self.cfg['her_state_size']:]

        x = x.view(x.size(0), self.cfg['history_count'], -1)

        history = history.view(1, state.size(0), -1)
        out, hidden = self.rnn(x, history)
        self.features = out[:, 0, :].view(1, 1, -1)

        x = hidden.view(state.size(0), -1)
        if self.cfg['her_state_size']:
            her_state = F.tanh(self.her_state(
                            state[:, :self.cfg['her_state_size']]))
            x = torch.cat([x, her_state], dim=1)

        return self.wrap_action(self.ex(x))

    def sample_noise(self):
        self.ex.sample_noise()

    def remove_noise(self):
        return
        self.ex.remove_noise()

    def recomputable(self):
        return True#False#

    def extract_features(self, states):
#        return None
        states = states[:, self.cfg['her_state_size']:self.cfg['her_state_size']+self.state_size:]
        states = torch.from_numpy(states)

        hidden = torch.zeros(1, 1, self.cfg['history_features'])

# GRU is faster than GRUCell on whole episode, with GRU we can speed it up here
# as it expose its whole state ( out_n = hidden_n ... )
        out, _ = self.rnn(states.unsqueeze(0), hidden)
        features = [h.reshape(1, 1, -1).detach().cpu().numpy() for h in out.squeeze(0)]
        ret = [np.zeros(shape=features[-1].shape)] + features[:-1]
        return ret

        for x in states:
            x = x.view(1, 1, -1)
            _, hidden = self.rnn(x, hidden)
            features.append(hidden.detach().cpu().numpy())

        return features
