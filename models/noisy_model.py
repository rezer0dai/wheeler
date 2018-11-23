import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.policy import *
from utils.nes import NoisyNet

# following paper : https://arxiv.org/pdf/1706.10295.pdf

class Actor(nn.Module):
    def __init__(self, state_size, action_size, wrap_action, cfg, hiddens = [400, 300]):
        super().__init__()
        state_size = state_size + cfg['her_state_features']
        self.net = NoisyNet([state_size] + hiddens + [action_size])
        self.algo = DDPG(wrap_action) if cfg['ddpg'] else PPO(action_size)

    def forward(self, goal, state):
        if goal.dim() == state.dim():
            state = torch.cat([goal, state], -1)
        x = self.net(state)
        return self.algo(x)

    def sample_noise(self):
        self.net.sample_noise()
    def remove_noise(self):
        return

class FeaturedActor(nn.Module):
    def __init__(self, _, action_size, wrap_action, cfg, hiddens = [400, 300]):
        super().__init__()

        assert not cfg['full_rnn_out'], "currently this is not supported ( full_rnn_out ) with FeaturedActor at noisynet"

        self.state_size = cfg['history_features']
        self.state_size = self.state_size // cfg['rnn_n_layers']
        self.state_size += cfg['her_state_features']

        self.net = NoisyNet([self.state_size] + hiddens + [action_size])
        self.algo = DDPG(wrap_action) if cfg['ddpg'] else PPO(action_size)

    def forward(self, goal, state):
        state = state[:, -self.state_size:] # extract only features !!
        if goal.dim() == state.dim():
            state = torch.cat([goal, state], -1)
        x = self.net(state)
        return self.algo(x)

    def sample_noise(self):
        self.net.sample_noise()
    def remove_noise(self):
        return
