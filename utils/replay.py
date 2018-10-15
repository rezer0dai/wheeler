import random, sys
import numpy as np

sys.path.append("PrioritizedExperienceReplay")
from PrioritizedExperienceReplay.proportional import Experience as Memory

from baselines.common.schedules import LinearSchedule

class ReplayBuffer:
    def __init__(self, cfg, objective_id, actor, update_goal = None):
        self.cfg = cfg

        assert None != update_goal or cfg['her_max_ratio'] == 0, "we dont update goals in non-HER games, therefore her_max_ratio must be 0 or her_state_size must be > 0"

        self.actor = actor

        self.objective_id = objective_id # identifier for fun_reward ~ our objective function id basically
        self.update_goal = update_goal

        self.inds = None

        self.count = 0
        self.beta = LinearSchedule(cfg['replay_beta_iters'],
               initial_p=cfg['replay_beta_base'],
               final_p=cfg['replay_beta_top'])
        self.mem = Memory(cfg['replay_size'], cfg['batch_size'], cfg['replay_alpha'])

    def sample(self, batch_size):
        self.inds, data = zip(*self._sample(batch_size))
# lol TODO : kick off numpy vstack transpose
        data = np.vstack(data)
#        print("-> new sample :", np.hstack(self.inds).shape, [i[0] for i in self.inds], data.shape)
        return data.T

    def add(self, batch, prios):
# if task is so hard to not do at least n_step per episode, then rethink different strategy ...
# .. learning from semi-professional sampled episode, and use those as trampoline
        if len(prios) < self.cfg['n_step'] * 2:
            return# this is due to our HER-sampling logic
# can be interesting to filter what to remember now ~ by curiosity sounds interesting to me
        if not self._worth_experience(prios):
            return

        for i, data in enumerate(batch):
            self.mem.add([data, i, len(prios) - i - 1], prios[i])

    def _worth_experience(self, prios):
        if not len(self):
            return True
        if len(self) < self.cfg['replay_size']:
            return True
        for _ in range(10):
            _, w, _ = self.mem.select(1.)
            if 0 == prios.mean():
                break
            status = (1. / prios.mean()) > (np.mean(w) * self.cfg['replay_size'])
            if status:
                return True
        return 0 == random.randint(0, 4)

    def _sample(self, batch_size):
        for _ in range(1):#batch_size):#
            self.count += 1
            batch, _, inds = self.mem.select(self.beta.value(self.count))
            data, local_forward, local_backward = zip(*batch)

# due to following uniq -> final batch will be most likely smaller than batch_size from config
# therefore, adjust batch_size in config to reflect that .. avoiding to add here some approximations
            uniq = set(map(lambda i_b: i_b[0] - i_b[1], zip(inds, local_forward)))
            for i, b, f in zip(inds, local_backward, local_forward):
                pivot = i - f
                if pivot < 0 or pivot + b + f > len(self):
                    continue # temporarely we want to avoid this corner case .. TODO
                if pivot not in uniq:
                    continue
                uniq.remove(pivot)
#                yield (i, self.mem.tree.data[i][0])
#                continue
                yield zip(*self._her_sample(pivot, b + f))

    def _her_sample(self, pivot, length):
        # TODO fix - n_step form max_count and avaiable_range !!
        # that is hotfix because later in her sampling we go out of range in tree..
        max_count = min(self.cfg['max_ep_draw_count'], length - self.cfg['n_step'])
        available_range = range(length - self.cfg['n_step'])
        replay = random.sample(available_range, random.randint(1, max_count))

        episode = self.actor.extract_features( # python hopefully passing this as a reference only..
                self.mem.tree.data[pivot:pivot+length], self.cfg['n_step'], replay)

        for i, e in zip(replay, episode):
            step = e if 0 == random.randint(
                    0, self.cfg['her_max_ratio']) else self.update_goal(e,
                        self.mem.tree.data[pivot + i + random.randint(0,
                            self.cfg['n_step'] + random.randint(0, 1) * (length - self.cfg['n_step'] - i))][0][0],
                        np.vstack(self.mem.tree.data[pivot + i : pivot + i + self.cfg['n_step']])[:, 0],
                        self.objective_id, self.cfg['discount_rate'])
            yield pivot + i, step

    def update(self, prios):
        self.mem.priority_update(np.hstack(self.inds), prios)
        self.inds = None

    def __len__(self):
        return self.mem.tree.filled_size()
