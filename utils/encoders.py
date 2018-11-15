import abc
import numpy as np

from utils.rbf import *
from utils.normalizer import *

class IEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def out_size(self):
        pass

    @abc.abstractmethod
    def forward(self, states, history):
        pass

    def extract_features(self, states):
        feats = np.zeros(len(states), 1, 1, self.cfg['history_features'])
        return self.forward(states, feats)

class StackedEncoder(IEncoder):
    def __init__(self, size_in, encoder_a, encoder_b):
        super().__init__()
        self.size = size_in
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
    def out_size(self):
        return self.encoder_b.out_size()
    def forward(self, states, history):
        size = states.size(0)
        states, history = self.encoder_a(states.reshape(-1, self.size), history)
        states = states.reshape(size, -1)
        return self.encoder_b(states, history)

class RBFEncoder(IEncoder):
    def __init__(self, env, gamas, components, sampler = None):
        super().__init__()
        self.size = len(env.reset())
        self.encoder = RbfState(env, gamas, components, sampler)
    def out_size(self):
        return self.encoder.out_size()
    def forward(self, states, history):
        size = states.size(0)
        states = self.encoder.transform(states.reshape(-1, self.size))
        return torch.from_numpy(states.reshape(size, -1)), history

class BatchNormalizer2D(IEncoder):
    def __init__(self, cfg, state_size):
        super().__init__()
        self.osize = state_size
        self.size = cfg['her_state_size'] + state_size * cfg['history_count']
        self.bn = nn.BatchNorm1d(self.size)
    def out_size(self):
        return self.osize
    def forward(self, states, history):
        if states.size(0) > 1:
            return self.bn(states), history
        self.eval() # this must be not called for training trolol
        out = self.bn(states)
        self.train()
        return out, history
class BatchNormalizer3D(IEncoder):
    def __init__(self, state_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(state_size)
        self.size = state_size
    def out_size(self):
        return self.size
    def forward(self, states, history):
        full_shape = states.shape
        states = states.reshape(states.size(0), self.size, -1)
        return self.bn(states).reshape(full_shape), history

class GlobalNormalizerWGrads(IEncoder):
    def __init__(self, state_size):
        super().__init__()
        self.bn = Normalizer(state_size)
    def out_size(self):
        return self.bn.size
    def forward(self, states, history):
        full_shape = states.shape
        states = states.reshape(-1, self.bn.size) # we stacking history states ( frames ) as well to batchnorm!
        self.bn.update(states)
        return self.bn.normalize(states).reshape(full_shape), history
class GlobalNormalizer(IEncoder):
    def __init__(self, state_size):
        super().__init__()
        self.bn = Normalizer(state_size)
    def out_size(self):
        return self.bn.size
    def forward(self, states, history):
        full_shape = states.shape
        states = states.reshape(-1, self.bn.size) # we stacking history states ( frames ) as well to batchnorm!
        self.bn.update(states)
        return self.bn.normalize(states).detach().reshape(full_shape), history
