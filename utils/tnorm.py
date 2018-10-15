"""
                    **** IMPORTANT ****

-> copy-pasted to my project from : https://github.com/vitchyr/rlkit
  >> encouraged to check that nice project
"""
import torch
import numpy as np

from torch.autograd import Variable

def np_to_var(np_array, **kwargs):
    return Variable(torch.from_numpy(np_array), **kwargs).double()

from utils.normalizer import Normalizer, FixedNormalizer

class TorchNormalizer(Normalizer):
    """
    Update with np array, but de/normalize pytorch Tensors.
    """
    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = np_to_var(self.mean, requires_grad=False)
        std = np_to_var(self.std, requires_grad=False)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean = np_to_var(self.mean, requires_grad=False)
        std = np_to_var(self.std, requires_grad=False)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std


class TorchFixedNormalizer(FixedNormalizer):
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = np_to_var(self.mean, requires_grad=False)
        std = np_to_var(self.std, requires_grad=False)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def normalize_scale(self, v):
        """
        Only normalize the scale. Do not subtract the mean.
        """
        std = np_to_var(self.std, requires_grad=False)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v / std

    def denormalize(self, v):
        mean = np_to_var(self.mean, requires_grad=False)
        std = np_to_var(self.std, requires_grad=False)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std

    def denormalize_scale(self, v):
        """
        Only denormalize the scale. Do not add the mean.
        """
        std = np_to_var(self.std, requires_grad=False)
        if v.dim() == 2:
            std = std.unsqueeze(0)
        return v * std
