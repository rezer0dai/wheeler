import numpy as np
import random
from collections import deque

class Memory(object):
    def __init__(self, cfg):
        self.memory = deque(maxlen=cfg['replay_size'])

    def add(self, batch, _, __):
        for data in batch:
            self.memory.append(data)

    def sample(self, batch_size, _):
        return (np.vstack(random.sample(self.memory, k=batch_size)).T)

    def update(self, _):
        pass

    def reset_alpha(self, _):
        pass

    def __len__(self):
        return len(self.memory)
