import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nes import NoisyLinear

def initialize_weights(layer):
    if type(layer) not in [nn.Linear, ]:
        return
    nn.init.kaiming_uniform_(layer.weight)

class CriticNN(nn.Module):

    def __init__(self, task, cfg):
        super(CriticNN, self).__init__()
        self.wrap_value = task.wrap_value

        self.cfg = cfg

        self.state_dim = cfg['her_state_size'] + task.state_size() * cfg['history_count'] + task.action_size()
        self.action_dim = task.action_size()

        self.fcs1 = nn.Linear(self.state_dim,256,bias=False)
        self.fcs2 = nn.Linear(256,128)

        self.fca1 = nn.Linear(self.action_dim,128,bias=False)

        self.fc2 = nn.Linear(256,128)

        self.fc3 = nn.Linear(128,1,bias=False)
        self.apply(initialize_weights)

    def forward(self, state, action, _):
        x = torch.cat([state, action], 1)
        s1 = F.relu(self.fcs1(x))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2,a1),dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.wrap_value(x)

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

        self.fc1 = nn.Linear(self.state_dim,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = NoisyLinear(128,64)
        self.fc4 = NoisyLinear(64,self.action_dim)
        self.apply(initialize_weights)

    def forward(self, state, _):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        self.features = torch.zeros(state.size(0), 1, self.cfg['history_features'])
        return self.wrap_action(self.fc4(x))

    def sample_noise(self):
        self.fc4.sample_noise()
        self.fc3.sample_noise()

    def remove_noise(self):
        return

    def recomputable(self):
        return False
