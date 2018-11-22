import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.nes import *

# well just POC of my mindset how curiosity should works ( not properly tested imho .. )

class CuriosityNN(nn.Module):
    def __init__(self, state_size, action_size, cfg, wrap_action):
        super(CuriosityNN, self).__init__()
        torch.set_default_tensor_type(cfg['tensor'])
        self.cfg = cfg

        self.state_size = state_size * cfg['history_count']
        self.action_size = action_size
        self.wrap_action = wrap_action

#        self.net = NoisyNet([self.state_size * 2, 64, 64, self.action_size])
        self.net = NoisyNet([self.state_size * 2, 64, self.action_size]) # do it easy weight for tests

    def forward(self, state, next_state):
        state = torch.DoubleTensor(state).to(self.cfg['device'])
        next_state = torch.DoubleTensor(next_state).to(self.cfg['device'])
        flow = torch.cat([ state, next_state ], dim=1)
        return self.wrap_action(
                self.net(flow))

    def renoise(self):
        self.net.sample_noise()

    def parameters(self):
        return self.net.parameters()

class CuriosityPrio:
    def __init__(self, state_size, action_size, action_range, wrap_action, device, cfg):
        self.cfg = cfg
        self.rewarder = CuriosityNN(state_size, action_size, cfg, wrap_action).to(device)
        self.action_range = action_range
        self.opt = torch.optim.Adam(self.rewarder.parameters(), 1e-4)

    def weight(self, s, n, a):
        if not self.cfg['use_curiosity_buf']:
            return np.abs(np.random.randn(len(a)))
        action = self.rewarder(s, n).detach()
        diff = (a - action)
        norm = np.divide(diff, self.action_range)
        dist = np.abs(norm)
        clip = np.clip(dist, 0. + 1e-10, 1. - 1e-10)
        scale = clip ** 2
        reward = scale.mean(dim=1).cpu().numpy()
        return reward

    def update(self, s, n, a):
        if not self.cfg['use_curiosity_buf']:
            return
        a = torch.DoubleTensor(a).reshape(len(a), -1).to(self.cfg['device'])
        def optim():
            self.opt.zero_grad()
            loss = F.smooth_l1_loss(
                    self.rewarder(s, n), a)
            loss.backward()
        self.opt.step(optim)
        self.rewarder.renoise()
