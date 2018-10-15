import random, sys
from collections import namedtuple, deque
import numpy as np

sys.path.append("../../openai/baselines/")

from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import PrioritizedReplayBuffer

class Storage:
    def __init__(self, r, n_r, f, n_f):
        self.r = r
        self.n_r = n_r
        self.f = f
        self.n_f = n_f

class ReplayBuffer:
    def __init__(self, cfg, buffer_size):
        self.prb = PrioritizedReplayBuffer(buffer_size, cfg['replay_alpha'])

        self.beta = LinearSchedule(cfg['replay_beta_iters'],
               initial_p=cfg['replay_beta_base'],
               final_p=cfg['replay_beta_top'])

        self.count = 0
        self.prio_eps = cfg['replay_prio_eps']
        self.batch_idxes = None
        self.batch_weights = None

    def add(self, states, rewards, actions, features, n_states, n_rewards, n_features):
        self.count += len(rewards)
        for i, (s, r, a, f, n_s, n_r, n_f) in enumerate(zip(states, rewards, actions, features, n_states, n_rewards, n_features)):
            self.prb.add(s, a, Storage(r, n_r, f, n_f), n_s, i == len(rewards) - 1)

    def sample(self, batch_size):
        (s, a, r_nr_f_nf, n_s, _, self.batch_weights, self.batch_idxes) = self.prb.sample(batch_size, self.beta.value(self.count))
        # eh uh we dont want to modify openai code, so some temporal workaround :)
        r, n_r, f, n_f = zip(*map(lambda x: (x.r, x.n_r, x.f, x.n_f), r_nr_f_nf))
        return (s, r, a, f, n_s, n_r, n_f)

    def update(self, error_weights):
        assert self.batch_idxes, "seems unlocked updates on replay buffer ..."
        if not self.batch_idxes:
            return
        prios = np.abs(error_weights) + self.prio_eps
        self.prb.update_priorities(self.batch_idxes, prios)
        self.batch_idxes = None

    def __len__(self):
        return len(self.prb)
