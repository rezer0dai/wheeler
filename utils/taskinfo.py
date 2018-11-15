import abc
import torch

class TaskInfo:
    def __init__(self,
            state_size, action_size, action_low, action_high,
            cfg, # general config with task info
            replaybuf,
            factory, Mgr, args):

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high

        self.action_range = self.action_high - self.action_low

        self.cfg = cfg
        self.env = Mgr(factory, *args)
        self.replaybuf = replaybuf

    def wrap_value(self, x):
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.clamp(x, min=self.action_low, max=self.action_high)

    def make_replay_buffer(self, cfg):
        return self.replaybuf(cfg)

    @abc.abstractmethod
    def new(self, cfg, bot_id, objective_id):
        pass
