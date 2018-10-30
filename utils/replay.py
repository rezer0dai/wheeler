import random, sys
import numpy as np

sys.path.append("PrioritizedExperienceReplay")
from PrioritizedExperienceReplay.proportional import Experience as Memory

from baselines.common.schedules import LinearSchedule

class ReplayBuffer:
    def __init__(self, cfg, objective_id):
        self.cfg = cfg

        self.objective_id = objective_id # identifier for fun_reward ~ our objective function id basically

        self.inds = None

        self.count = 0
        self.beta = LinearSchedule(cfg['replay_beta_iters'],
               initial_p=cfg['replay_beta_base'],
               final_p=cfg['replay_beta_top'])
        self.mem = Memory(cfg['replay_size'], cfg['batch_size'], cfg['replay_alpha'])

    def sample(self, batch_size, critic):
#        self.inds, data = zip(*self._sample(batch_size, critic))
#        off_exc = '''
        try: # due to buf od priority replay github code i am using
            self.inds, data = zip(*self._sample(batch_size, critic))
        except:
            return None
#        '''
# lol TODO : kick off numpy vstack transpose
        data = np.vstack(data)
        return (data.T)

    def add(self, batch, prios, delta, hashkey):
# if task is so hard to not do at least n_step per episode, then rethink different strategy ...
# .. learning from semi-professional sampled episode, and use those as trampoline
        if len(prios) < self.cfg['n_step'] * 2:
            return
        if len(prios) < delta % self.objective_id:
            return
# can be interesting to filter what to remember now ~ by curiosity sounds interesting to me
        assert self.objective_id > 0, "objective id must be > 1 -> {}".format(self.objective_id)
        offset = ((delta % self.cfg['n_critics']) + self.objective_id - 1) % self.cfg['n_critics']
        o_prios = prios[offset::self.cfg['n_critics']] # need double check..
        if not self._worth_experience(o_prios):
            return

        for i, data in enumerate(batch): # we weight only with what we are working with
            self.mem.add(
                    [data, i, len(prios) - i - 1, delta, hashkey], 
                    0. if (offset + i) % self.cfg['n_critics'] else prios[i])

    def _worth_experience(self, prios):
        if not len(self):
            return True
        if len(self) < self.cfg['replay_size']:
            return True
        for _ in range(10):
            _, w, _ = self.mem.select(1.)
            if None == w:
                continue
            status = prios.mean() > np.mean(w)
            if status:
                return True
        return 0 == random.randint(0, 4)

    def _sample(self, batch_size, critic):
        count = 0
        while not count:
            batch, _, inds = self.mem.select(self.beta.value(self.count))
            data, local_forward, local_backward, delta, hashkey = zip(*batch)

            self.count += 1

# due to following uniq -> final batch will be most likely smaller than batch_size from config
# therefore, adjust batch_size in config to reflect that .. avoiding to add here some approximations
            uniq = set(map(lambda i_b: i_b[0] - i_b[1], zip(inds, local_forward)))
            for i, b, f, d, k in zip(inds, local_backward, local_forward, delta, hashkey):
                if count >= self.cfg['max_ep_draw_count']:
                    break
                pivot = i - f
                if pivot < 0 or pivot + b + f > len(self):
                    continue # temporarely we want to avoid this corner case .. TODO
                if pivot not in uniq:
                    continue
                uniq.remove(pivot)
#                yield (i, self.mem.tree.data[i][0])
#                continue
                count += 1
                yield zip(*self._do_sample_wrap(pivot, b + f, d, critic, k))

    def _do_sample_wrap(self, pivot, length, delta, critic, hashkey):
        return self._do_sample(self.mem.tree.data[pivot:pivot+length], pivot, length, delta, critic, hashkey)

    def _do_sample(self, full_episode, pivot, length, delta, critic, _):
        available_range = range(
                ((delta % self.cfg['n_critics']) + self.objective_id - 1) % self.cfg['n_critics'],
                length, 
                self.cfg['n_critics'] if self.cfg['disjoint_critics'] else 1)

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
        return self.mem.tree.filled_size()
