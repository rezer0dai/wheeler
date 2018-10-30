# this we want to be as simple as possible, no additional logic nor assumptions
# just sharing what we saw .. replay buffs are forwarding us interestings stuffs
# sample therefore random
import random
from multiprocessing import Queue, Process
from utils.replay import *

class ExperienceObserver(Process):
    def __init__(self, cfg):
        super().__init__()

        self.exps = [None] * cfg['cross_exp_size']
        self.keys = {}
        self.ind = 0

        self.cmd = { "add" : self._add, "sample" : self._sample }

        self.channel = Queue()
        self.sampler = Queue()

    def run(self):
        while True: # single thread is fine
            data = self.channel.get()
            cmd, data = data
            self.cmd[cmd](*data)

    def add(self, data, hashkey):
        self.channel.put(("add", (data, hashkey)))

    def sample(self):
        self.channel.put(("sample", ()))
        return self.sampler.get()

    def _add(self, data, key):
        if key in self.keys:
            return
        self.keys[key] = self.ind
        self.exps[self.ind] = data
        self.ind = (self.ind + 1) % len(self.exps)

    def _sample(self):
        data = random.choice(self.exps[:len(self.keys)]) if len(self.keys) else None
        self.sampler.put(data)

class CrossExpBuffer(ReplayBuffer):
    def __init__(self, mgr, cfg, objective_id):
        super().__init__(cfg, objective_id)
        self.mgr = mgr

    def _do_sample(self, full_episode, pivot, length, delta, critic, hashkey):
        # forwarding already sampled data
        for i, data in super()._do_sample(full_episode, pivot, length, delta, critic, hashkey):
            yield i, data

        # withdraw possibly cross-sampled data ( cross simulation / cross bot )
        cross_episode = self.mgr.sample()

# registering hot data to observer ~ or we can add them when adding to buffers
# .. but currently add data based on heatmap ( how often are used ) seems to me ok idea
        self.mgr.add(full_episode, hashkey)

        if None == cross_episode:
            return # nothin to be seen yet

        for _, data in super()._do_sample(
                cross_episode, 
                0, len(cross_episode), 
                random.randint(0, self.cfg['n_critics']), critic, hashkey):
            yield -1, data # indicating we dont want to touch that data at update prios later on

    def update(self, prios):
        self.inds = np.hstack(self.inds).reshape(-1)
#        assert len(prios) == sum(map(lambda i: i >= 0, self.inds)), "FAIL" # good
        prios = prios[self.inds >= 0]
        self.inds = self.inds[self.inds >= 0]
        assert len(prios) == sum(map(lambda i: i >= 0, self.inds)), "FAIL" # nope
        super().update(prios)

def cross_exp_buffer(cfg):
    mgr = ExperienceObserver(cfg)
    def cross_buff(cfg, objective_id):
        ceb = CrossExpBuffer(mgr, cfg, objective_id)
        return ceb

    mgr.start() # separate process as we overhauling our main process with python threads 
    return cross_buff
