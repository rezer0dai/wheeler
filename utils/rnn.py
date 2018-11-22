from utils.encoders import IEncoder

import numpy as np

import torch
import torch.nn as nn

# lots of code duplexity .. generalize it, also add vanila RNN
# also can be improved for SPEED ~ we dont need to manually loop in extract features ( sequence is same )
# same time, we dont need to recalc features everytime ( only if too old based on replaybuffer stats )

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
        x = states.view(states.size(0), self.history_count, -1)

        history = history.view(self.n_layers, states.size(0), -1)
        out, hidden = self.rnn(x, history)

        if self.out_count < self.history_count:
            return out[:,-1,:].view(states.size(0), -1), hidden.reshape(states.size(0), 1, -1)
        return out.contiguous().view(states.size(0), -1), hidden.reshape(states.size(0), 1, -1)

    def extract_features(self, states):
        outs = []
        features = []
        hidden = torch.zeros(self.n_layers, 1, self.n_features)
        for state in states:
            features.append(hidden.detach().cpu().numpy().reshape(1, 1, -1))

            state = state.view(1, self.history_count, -1)

            out, hidden = self.rnn(state, hidden)

            if self.out_count < self.history_count:
                outs.append(out[:,-1:])
            else:
                outs.append(out)

        return torch.stack(outs).view(states.size(0), -1).detach(), features

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
        x = states.view(states.size(0), self.history_count, -1)

        history = history.view(2, self.n_layers, states.size(0), -1)
        out, hidden = self.rnn(x, history)

        if self.out_count < self.history_count:
            return out[:,-1,:].view(states.size(0), -1), torch.cat(hidden).reshape(states.size(0), 1, -1)
        return out.contiguous().view(states.size(0), -1), torch.cat(hidden).reshape(states.size(0), 1, -1)

    def extract_features(self, states):
        outs = []
        features = []
        hidden = [ torch.zeros(self.n_layers, 1, self.n_features) ] * 2

        for state in states:
            features.append(torch.cat(hidden).detach().cpu().numpy().reshape(1, 1, -1))

            state = state.view(1, self.history_count, -1)

            out, hidden = self.rnn(state, hidden)

            if self.out_count < self.history_count:
                outs.append(out[:,-1:])
            else:
                outs.append(out)

        return torch.stack(outs).view(states.size(0).detach(), -1), features

class GruLayer(nn.Module):
    def __init__(self, in_count, out_count, bias):
        super().__init__()
        self.net = nn.GRU(
                in_count, out_count,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=bias)
    def forward(self, states, hidden):
        return self.net(states, hidden)

class FasterGRUEncoder(IEncoder):
    """
    most likely we can not do this with LSTM
    as GRU provide us full hidden state ( in out ), while LSTM only last one for t~seq_len
    - this is problematic to extract features in fast fashion ( 1 forward pass )

    - also we banned now full RNN state forwarding; as in extract features will need
      more complicated design ( stacking outputs together.. now not worth implement ~ overengineering )
    """
    def __init__(self, cfg, state_size):
        super().__init__({ 'history_count' : 1, 'history_features' : cfg['history_features'] })

        assert not cfg['full_rnn_out'], "faster GRU now support only output forwarding! fix your full_rnn_out"

        self.n_layers = cfg['rnn_n_layers']
        assert 0 == cfg['history_features'] % self.n_layers, "history_features must be fraction of rnn_n_layers!"

        self.n_features = cfg['history_features'] // self.n_layers
        self.history_count = cfg['history_count']
        self.state_size = state_size // self.history_count

        self.rnn = [ GruLayer(
            self.state_size if not i else self.n_features, self.n_features, bool(i + 1 < self.n_layers)
            ) for i in range(self.n_layers) ]

        for i, rnn in enumerate(self.rnn):
            self.add_module("fast_gru_%i"%i, rnn)

    def has_features(self):
        return True

    def out_size(self):
        return self.n_features

    def forward(self, states, history):
        memory = self._forward_impl(states, history)

        out = memory[-1, :, -1, :] # last layer, all states, final output, all features
        # if we want to go all out, then following line is way to go
        #  out = memory[-1, :, :, :].view(states.size(0), -1)

        # all layers, all states, first hidden state, all features
        hidden = memory[:, :, 0, :].transpose(0, 1).reshape(states.size(0), 1, -1)
        return out, hidden

    def extract_features(self, states):
        hidden = torch.zeros(self.n_layers, 1, self.n_features)
        sequence = torch.cat([
            states[0].reshape(self.history_count, self.state_size), # fist state compose of initial states as well for sequence
            states[1:, -self.state_size:]]).unsqueeze(0) # later on extract actual state from sequence

        memory = self._forward_impl(sequence, hidden)

        out = memory[-1, 0, self.history_count-1:, :] # last layer, only one state, whole history sequence of features
        hidden = memory[:, 0, self.history_count-1:, :, ].transpose(0, 1).reshape(states.size(0), 1, 1, -1)
        return out.detach(), hidden.detach().cpu().numpy()

    def _forward_impl(self, states, history):
        out = states.view(states.size(0), -1, self.state_size)
        history = history.view(self.n_layers, states.size(0), -1)

        memory = []
        for rnn, feats in zip(self.rnn, history):
            out, _ = rnn(out, feats.unsqueeze(0))
            memory.append(out)

        return torch.stack(memory)
