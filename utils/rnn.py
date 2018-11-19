from utils.encoders import IEncoder

import numpy as np
from collections import deque

import torch
import torch.nn as nn

# lots of code duplexity .. generalize it, also add vanila RNN

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

        self.n_layers = cfg['rnn_n_layers']
        assert 0 == cfg['history_features'] % self.n_layers, "history_features must be fraction of rnn_n_layers!"

        self.her_state = cfg['her_state_size']
        self.n_features = cfg['history_features'] // self.n_layers
        self.history_count = cfg['history_count']
        self.state_size = state_size // self.history_count

        self.out_count = self.history_count if cfg['full_rnn_out'] else 1

        self.rnn = nn.GRU(
                self.state_size, self.n_features,
                batch_first=True,
                bidirectional=False,
                dropout=.2,
                num_layers=self.n_layers,
                bias=False)

    def has_features(self):
        return True

    def out_size(self):
        return self.n_features * self.out_count

    def forward(self, states, history):
        x = states[:, self.her_state:]
        x = x.view(x.size(0), self.history_count, -1)

        history = history.view(self.n_layers, states.size(0), -1)
        out, hidden = self.rnn(x, history)

        if self.out_count < self.history_count:
            return out[:,-1,:].view(states.size(0), -1), hidden.reshape(1, -1)
        return out.contiguous().view(states.size(0), -1), hidden.reshape(1, -1)

    def extract_features(self, states):
        states = states[:, # we want to get context from before our state
                self.her_state:self.her_state+self.state_size:]

        history = deque(maxlen=self.history_count)
        for s in [torch.zeros(len(states[0]))] * self.history_count:
            history.append(s)

        outs = []
        features = []
        hidden = torch.zeros(self.n_layers, 1, self.n_features)
        for s in states:
            features.append(hidden.detach().cpu().numpy().reshape(1, -1))

            history.append(s)
            state = torch.stack([s for s in history]).squeeze(1)
            state = state.view(1, self.history_count, -1)

            out, hidden = self.rnn(state, hidden)

            if self.out_count < self.history_count:
                outs.append(out[:,-1:])
            else:
                outs.append(out)

        return torch.stack(outs).view(states.size(0), -1), features

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

        self.n_layers = cfg['rnn_n_layers']
        assert 0 == cfg['history_features'] % self.n_layers, "history_features must be fraction of rnn_n_layers!"

        self.her_state = cfg['her_state_size']
        self.n_features = cfg['history_features'] // (2 * self.n_layers)
        self.history_count = cfg['history_count']
        self.state_size = state_size // self.history_count

        self.out_count = self.history_count if cfg['full_rnn_out'] else 1

        self.rnn = nn.LSTM(
                self.state_size, self.n_features,
                batch_first=True,
                bidirectional=False,
                dropout=.2,
                num_layers=self.n_layers,
                bias=False)

    def has_features(self):
        return True

    def out_size(self):
        return self.n_features * self.out_count

    def forward(self, states, history):
        x = states[:, self.her_state:]
        x = x.view(x.size(0), self.history_count, -1)

        history = history.view(2, self.n_layers, states.size(0), -1)
        out, hidden = self.rnn(x, history)

        if self.out_count < self.history_count:
            return out[:,-1,:].view(states.size(0), -1), torch.cat(hidden).reshape(1, -1)
        return out.contiguous().view(states.size(0), -1), torch.cat(hidden).reshape(1, -1)

    def extract_features(self, states):
        states = states[:, # we want to get context from before our state
                self.her_state:self.her_state+self.state_size:]

        history = deque(maxlen=self.history_count)
        for s in [torch.zeros(len(states[0]))] * self.history_count:
            history.append(s)

        outs = []
        features = []
        hidden = [torch.zeros(self.n_layers, 1, self.n_features), torch.zeros(self.n_layers, 1, self.n_features)]

        for s in states:
            features.append(torch.cat(hidden).detach().cpu().numpy().reshape(1, -1))

            history.append(s)
            state = torch.stack([s for s in history]).squeeze(1)
            state = state.view(1, self.history_count, -1)

            out, hidden = self.rnn(state, hidden)

            if self.out_count < self.history_count:
                outs.append(out[:,-1:])
            else:
                outs.append(out)

        return torch.stack(outs).view(states.size(0), -1), features