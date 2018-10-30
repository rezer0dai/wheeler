import numpy as np
import toml, threading
from unityagents import UnityEnvironment
from torch.multiprocessing import SimpleQueue

# all this runs on server ~ main process ~ side ( as we want to proxy it trough LocalTaskManager )

# likely we want to scope those globals to some singleton class
CFG = toml.loads(open('cfg.toml').read())
ENV = UnityEnvironment(file_name=CFG['task'])
BRAIN_NAME = ENV.brain_names[0]
def _reset(cfg):#config=cfg[0], 
    return ENV.reset(train_mode=True)[BRAIN_NAME]
def _step(action):
    return ENV.step(np.asarray(action).reshape(-1))[BRAIN_NAME]
CONTROL = { "reset" : _reset, "step" : _step }

class UnityBrain:
    def __init__(self, barier, ind):
        self.barier = barier
        self.ind = ind - 1
        self.cfg = None

    def seed(self, cfg):
        self.cfg = cfg

    def reset(self):
        cmd = "reset"
        env_info = CONTROL[cmd]([self.cfg]) if self.ind == -1 else self.barier.invoke(
                self.ind, cmd, self.cfg)

        state = env_info.vector_observations[self.ind]
        return state

    def step(self, action):
        cmd = "step"
        env_info = CONTROL[cmd](action) if self.ind == -1 else self.barier.invoke(
                self.ind, cmd, action)

        state = env_info.vector_observations[self.ind]
        done = env_info.local_done[self.ind]
        reward = env_info.rewards[self.ind]
        return (state, done, reward)

    def render(self):
        pass

class UnityGroupSync:
    def __init__(self, num):
        self.lock = threading.RLock()
        self.pipes = [SimpleQueue() for _ in range(num)]
        self.counter = { "reset" : [], "step" : [] }

    def _pass_trough(self, ind, cmd, data):
        with self.lock:
            self.counter[cmd] += [data]
            if len(self.counter[cmd]) != len(self.pipes):
                return False
        return True
        
    def _barier(self, ind, cmd, data):
        if not self._pass_trough(ind, cmd, data):
            return

        data = self.counter[cmd]
        self.counter[cmd] = []
        out = CONTROL[cmd](data)
        for p in self.pipes:
            p.put(out)

    def invoke(self, ind, cmd, data):
        self._barier(ind, cmd, data)
        data = self.pipes[ind].get()
        return data

def unity_factory(count):
    barier = UnityGroupSync(count)
    def factory(ind):
        return UnityBrain(barier, ind)
    return factory
