import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CuriosityNN(nn.Module):
    def __init__(self, task, cfg, wrap_action):
        super(CuriosityNN, self).__init__()
        torch.set_default_tensor_type(cfg['tensor'])
        self.cfg = cfg

        self.state_size = cfg['her_state_size'] + task.state_size() * self.cfg['history_count']
        self.action_size = task.action_size()
        self.wrap_action = wrap_action

        self.net = nn.Sequential(
                    nn.Linear(self.state_size * 2, 64),
                    nn.ReLU(True),
                    nn.Linear(64, self.action_size),
                    )

    def forward(self, state, next_state):
        state = torch.DoubleTensor(state).to(self.cfg['device'])
        next_state = torch.DoubleTensor(next_state).to(self.cfg['device'])
        flow = torch.cat([ state, next_state ], dim=1)
        return self.wrap_action(
                self.net(flow))

class CuriosityPrio:
    def __init__(self, task, cfg):
        self.cfg = cfg
        self.rewarder = CuriosityNN(task, cfg, task.wrap_action).to(task.device())
        self.action_range = task.action_range()
        self.opt = torch.optim.Adam(self.rewarder.parameters(), 1e-3)

    def weight(self, s, n, a):
        action = self.rewarder(s, n).detach()
        diff = (a - action)
        norm = np.divide(diff, self.action_range)
        dist = np.abs(norm)
        clip = np.clip(dist, 0. + 1e-10, 1. - 1e-10)
        scale = clip ** 2
        raw = scale.mean(dim=2).view(clip.size(0), -1)
        return raw.mean(dim=1)

    def update(self, s, n, a):
        a = torch.DoubleTensor(a).reshape(len(a), -1).to(self.cfg['device'])
        def optim():
            self.opt.zero_grad()
            loss = F.smooth_l1_loss(
                    self.rewarder(s, n), a)
            loss.backward()
        self.opt.step(optim)
