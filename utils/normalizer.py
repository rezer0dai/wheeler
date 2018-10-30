"""
Based on code from Marcin Andrychowicz

                    **** IMPORTANT ****

-> copy-pasted to my project from : https://github.com/vitchyr/rlkit
  >> encouraged to check that nice project
"""
import numpy as np
import torch

class Normalizer(torch.nn.Module):
    def __init__(
            self,
            size,
            eps=1e-8,
            default_clip_range=np.inf,
            mean=0,
            std=1,
    ):
        super().__init__()
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        self.sum = torch.nn.Parameter(torch.zeros(self.size))
        self.sumsq = torch.nn.Parameter(torch.zeros(self.size))
        self.count = torch.nn.Parameter(torch.ones(1))
        self.mean = torch.nn.Parameter(mean + torch.zeros(self.size))
        self.std = torch.nn.Parameter(std * torch.ones(self.size))

        self.synchronized = True

    def update(self, v):
        if v.ndim == 1:
            v = np.expand_dims(v, 0)
        assert v.ndim == 2
        assert v.shape[1] == self.size
        self.sum.data = self.sum.data + torch.tensor(v.sum(axis=0))
        self.sumsq.data = self.sumsq.data + torch.tensor(np.square(v).sum(axis=0))
        self.count[0] = self.count[0] + v.shape[0]
        self.synchronized = False

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self._synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range

        # convert back to numpy ( torch is just for sharing data between workers )
        std = self.std.detach().numpy()
        mean = self.mean.detach().numpy()

        if v.ndim == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return np.clip((v - mean) / std, -clip_range, clip_range)

    def _synchronize(self):
        self.mean.data = self.sum.detach() / self.count[0]
        self.std.data = torch.sqrt(
            np.maximum(
                np.square(self.eps),
                self.sumsq.detach() / self.count.detach()[0] - (self.mean.detach() ** 2)
            )
        )
        self.synchronized = True

class IdentityNormalizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def update(self, v):
        pass

    def normalize(self, v, clip_range=None):
        return v
