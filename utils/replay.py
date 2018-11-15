import random, sys
import numpy as np

sys.path.append("PrioritizedExperienceReplay")
from PrioritizedExperienceReplay.proportional import Experience as Memory

from utils.learning_rate import LinearAutoSchedule as LinearSchedule

class ReplayBuffer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.inds = None

        self.beta = LinearSchedule(cfg['replay_beta_iters'],
               initial_p=cfg['replay_beta_base'],
               final_p=cfg['replay_beta_top'])

        self.mem = Memory(cfg['replay_size'], cfg['batch_size'], cfg['replay_alpha'])

    def sample(self, batch_size, critic):
        self.inds, data = zip(*self._sample(batch_size, critic))
# lol TODO : kick off numpy vstack transpose
        data = np.vstack(data)
        return (data.T)

    def add(self, batch, prios, hashkey):
        if len(prios) < self.cfg['n_step'] * 2:
            return
        if not self._worth_experience(prios):
            return

        for i, data in enumerate(batch):
            self.mem.add([data, i, len(prios) - i - 1, hashkey], prios[i])

    def _worth_experience(self, prios):
        if not len(self):
            return True
        if len(self) < self.cfg['replay_size']:
            return True
        for _ in range(10):
            data = self.mem.select(1.)
            if None == data:
                return True
            _, w, _ = data
            status = prios.mean() > np.mean(w)
            if status:
                return True
        return 0 == random.randint(0, 4)

    def _sample(self, batch_size, critic):
        count = 0
        while not count:
            data = self.mem.select(self.beta.value())
            if None == data:
                continue
            batch, _, inds = data
            data, local_forward, local_backward, hashkey = zip(*batch)

            uniq = set(map(lambda i_b: i_b[0] - i_b[1], zip(inds, local_forward)))
            for i, b, f, k in zip(inds, local_backward, local_forward, hashkey):
#                if count >= self.cfg['max_ep_draw_count']:
#                    break
                pivot = i - f
                if pivot < 0 or pivot + b + f > len(self):
                    continue # temporarely we want to avoid this corner case .. TODO
                if pivot not in uniq:
                    continue
                uniq.remove(pivot)
#                yield (i, self.mem.tree.data[i][0])
#                continue
                count += 1
                yield zip(*self._do_sample_wrap(pivot, b + f, critic, k))

    def _do_sample_wrap(self, pivot, length, critic, hashkey):
        return self._do_sample(self.mem.tree.data[pivot:pivot+length], pivot, length, critic, hashkey)

    def _do_sample(self, full_episode, pivot, length, critic, _):
        available_range = range(length)

        top = min(len(available_range), self.cfg['max_ep_draw_count'])
        replay = random.sample(available_range, random.randint(1, top))

        if not critic or not self.cfg['replay_reanalyze']:
            episode = map(lambda i: full_episode[i][0], replay)
        else:
            episode = critic.reanalyze_experience(full_episode, replay)

        for i, step in zip(replay, episode):
            yield pivot + i, step

    def update(self, prios):
        '''
        replay buffer must be single thread style access, or properly locked ...
          ( sample, update, add )
          well in theory as it is not expanding, we dont care much of reads only .. for now lol ..
        '''
        self.mem.priority_update(np.hstack(self.inds), prios)
        self.inds = None

    def __len__(self):
        return len(self.mem)
