from utils.encoders import IEncoder

import numpy as np

import torch
import torch.nn as nn

# rethink multiple layered RNN; but dont increase much of complexity
# should be done easily, see GRU vs LSTM cheat on n_features ( // 2 for LSTM )
# so coupling hidden states after forward, and decouping it before passing to rnn

class GRUEncoder(IEncoder):
    def __init__(self, cfg, state_size):
        """
        state_size must be total size : state_size * history_count
         - therefore in case of encoder before, then encoder.total_size() is OK
         - in case we feed directly outputs from environment then len(env.reset()) * history_count

         - remember now RNN must be last layer, aka no normalization / encoding features afterwards!
           - unless in actor/critic network itself
        """
        super().__init__({ 'history_count' : 1, 'history_features' : cfg['history_features'] })
        self.her_state = cfg['her_state_size']
        self.n_features = cfg['history_features']
        self.history_count = cfg['history_count']
        self.state_size = state_size // self.history_count

        self.rnn = nn.GRU(
                self.state_size, self.n_features,
                batch_first=True,
                bidirectional=False,
#                dropout=.2,
                num_layers=1)

    def has_features(self):
        return True

    def out_size(self):
        return self.n_features

    def forward(self, states, history):
        x = states[:, self.her_state:]
        x = x.view(x.size(0), self.history_count, -1)

        history = history.view(1, states.size(0), -1)
        out, hidden = self.rnn(x, history)
        features = out[:, 0, :].view(1, 1, -1)

        return hidden.view(states.size(0), -1), features

    def extract_features(self, states):
        states = states[:, # we want to get context from before our state
                self.her_state:self.her_state+self.state_size:]

        hidden = torch.zeros(1, 1, self.n_features)

        out, _ = self.rnn(states.unsqueeze(0), hidden)
        features = [h.reshape(1, 1, -1).detach().cpu().numpy() for h in out.squeeze(0)]
        ret = [ hidden.numpy() ] + features[:-1]
        return torch.tensor(features).view(states.size(0), -1), ret

class LSTMEncoder(IEncoder):
    def __init__(self, cfg, state_size):
        """
        state_size must be total size : state_size * history_count
         - therefore in case of encoder before, then encoder.total_size() is OK
         - in case we feed directly outputs from environment then len(env.reset()) * history_count

         - remember now RNN must be last layer, aka no normalization / encoding features afterwards!
           - unless in actor/critic network itself
        """
        super().__init__({ 'history_count' : 1, 'history_features' : cfg['history_features'] })
        self.her_state = cfg['her_state_size']
        self.n_features = cfg['history_features'] // 2
        self.history_count = cfg['history_count']
        self.state_size = state_size // self.history_count

        self.rnn = nn.LSTMCell(
                self.state_size, self.n_features)

    def has_features(self):
        return True

    def out_size(self):
        return self.n_features

    def forward(self, states, hidden):
        x = states[:, self.her_state:]
        x = x.view(x.size(0), self.history_count, -1)

        out, feature = self.rnn(x.transpose(0, 1)[0, :], hidden.view(2, states.size(0), -1))
        features = torch.cat([out, feature], 1).view(1, 1, -1)
        for s in x.transpose(0, 1)[1:, :]:
            out, feature = self.rnn(s, (out, feature))

        return out.view(states.size(0), -1), features

    def extract_features(self, states):
        states = states[:, # we want to get context from before our state
                self.her_state:self.her_state+self.state_size:]

        hidden = torch.zeros(2, 1, self.n_features)
        feature = hidden.view(1, 1, -1)
        features = [feature.detach().cpu().numpy()]

        outs = []
        for x in states:
            x = x.view(1, -1)
            out, feature = self.rnn(x, hidden)
            hidden = (out, feature)
            feature = torch.cat(hidden, 1).view(1, 1, -1)

            outs.append(out.reshape(-1).detach().cpu().numpy())
            features.append(feature.detach().cpu().numpy())

        return torch.tensor(outs).view(len(states), -1), features[:-1]
