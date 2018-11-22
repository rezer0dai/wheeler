import abc
import numpy as np

import torch

from utils.rbf import *
from utils.normalizer import *

class IEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_history = cfg['history_count']
        self.n_features = cfg['history_features']

    @abc.abstractmethod
    def out_size(self):
        pass

    def count(self):
        return self.n_history

    def total_size(self):
        return self.out_size() * self.count()

    @abc.abstractmethod
    def forward(self, states, history):
        pass

    def has_features(self):
        return False

    def extract_features(self, states):
        feats = torch.zeros(len(states), 1, 1, self.n_features)
        return self.forward(states, feats)

# better to rethink design of this ~ beacuse of RNN ~ features, multiple ? dont over-engineer though...
class StackedEncoder(IEncoder):
    def __init__(self, cfg, size_in, encoder_a, encoder_b):
        super().__init__({ 'history_count' : 1, 'history_features' : cfg['history_features'] }) # well this is questionable .. rethink .. redo
        self.size = size_in
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        assert not self.encoder_a.has_features() or not self.encoder_a.has_features(), "only one RNN is allowed in encoder!"
        assert not self.encoder_a.has_features(), "Currently RNN can be only *last* layer of encoder!!"

    def out_size(self):
        return self.encoder_b.out_size()

    def has_features(self):
        return self.encoder_a.has_features() or self.encoder_a.has_features()

    def forward(self, states, history):
        size = states.size(0)
        states, history = self.encoder_a(states.reshape(-1, self.size), history)
        states = states.reshape(size, -1)
        return self.encoder_b(states, history)

    def extract_features(self, states):
        states, features_a = self.encoder_a.extract_features(states)
        states, features_b = self.encoder_b.extract_features(states)
        return states, features_b if self.encoder_b.has_features() else features_a

class IdentityEncoder(IEncoder):
    def __init__(self, cfg, size):
        super().__init__(cfg)
        self.size = size
    def out_size(self):
        return self.size
    def forward(self, states, history):
        return states, history

class RBFEncoder(IEncoder):
    def __init__(self, cfg, env, gamas, components, sampler = None):
        super().__init__(cfg)
        self.size = len(env.reset())
        self.encoder = RbfState(env, gamas, components, sampler)
    def out_size(self):
        return self.encoder.size
    def forward(self, states, history):
        size = states.size(0)
        states = self.encoder.transform(states.reshape(-1, self.size))
        return torch.from_numpy(states.reshape(size, -1)), history

class BatchNormalizer2D(IEncoder):
    def __init__(self, cfg, state_size):
        super().__init__(cfg)
        self.size = state_size * cfg['history_count']
        self.bn = nn.BatchNorm1d(self.size)
    def out_size(self):
        return self.size
    def forward(self, states, history):
        if states.size(0) > 1:
            return self.bn(states), history
        self.eval() # this must be not called for training trolol
        out = self.bn(states)
        self.train()
        return out, history
class BatchNormalizer3D(IEncoder):
    def __init__(self, cfg, state_size):
        super().__init__(cfg)
        self.bn = nn.BatchNorm1d(state_size)
        self.size = state_size
    def out_size(self):
        return self.size
    def forward(self, states, history):
        full_shape = states.shape
        states = states.reshape(states.size(0), self.size, -1)
        if states.size(0) > 1:
            return self.bn(states).reshape(full_shape), history
        self.eval() # this must be not called for training trolol
        out = self.bn(states).reshape(full_shape)
        self.train()
        return out, history

class GlobalNormalizerWGrads(IEncoder):
    def __init__(self, cfg, state_size):
        super().__init__(cfg)
        self.bn = Normalizer(state_size)
    def out_size(self):
        return self.bn.size
    def forward(self, states, history):
        full_shape = states.shape
        states = states.reshape(-1, self.bn.size) # we stacking history states ( frames ) as well to batchnorm!
        self.bn.update(states)
        return self.bn.normalize(states).reshape(full_shape), history
class GlobalNormalizer(GlobalNormalizerWGrads):
    def __init__(self, cfg, state_size):
        super().__init__(cfg, state_size)
        for p in self.bn.parameters():
            p.requires_grad = False
