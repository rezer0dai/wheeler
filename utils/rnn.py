from utils.encoders import IEncoder

import torch
import torch.nn as nn

class GRUEncoder(IEncoder):
    def __init__(self, state_size, her_state, history_count, n_features):
        super().__init__()
        self.state_size = state_size
        self.her_state = her_state
        self.history_count = history_count
        self.n_features = n_features

        self.rnn = nn.GRU(
                state_size, n_features,
                batch_first=True,
                bidirectional=False,
#                dropout=.2,
                num_layers=1)

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
        s = states[:, # we want to get context from before our state
                self.her_state:self.her_state_size+self.state_size:]

        hidden = torch.zeros(1, 1, self.n_features)

        out, _ = self.rnn(s.unsqueeze(0), hidden)
        features = [h.reshape(1, 1, -1).detach().cpu().numpy() for h in out.squeeze(0)]
        ret = [ hidden.numpy() ] + features[:-1]
        return states, ret
