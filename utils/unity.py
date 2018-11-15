import numpy as np
import threading, os
from unityagents import UnityEnvironment
from torch.multiprocessing import SimpleQueue

# all this runs on server ~ main process or server one ~ side
#       ( as we want to proxy it trough LocalTaskManager )

class UnityEnv:
    def __init__(self, cfg):
        self.env = UnityEnvironment(file_name=cfg['task'])
        self.brain_name = self.env.brain_names[0]

        self.ctrl = { "reset" : self._reset, "step" : self._step }

    def control(self, cmd, data):
        return self.ctrl[cmd](data)

    def ping(self):
        print("UNITY SERVER PROCESS PING ~ running at", os.getpid())

    def _reset(self, cfg):
        return self.env.reset(config=cfg, train_mode=True)[self.brain_name]

    def _step(self, action):
        return self.env.step(np.asarray(action).reshape(-1))[self.brain_name]

class UnityBrain:
    def __init__(self, env, barier, ind):
        self.env = env
        self.barier = barier
        self.ind = ind - 1
        self.cfg = None

    def seed(self, cfg):
        self.cfg = cfg

    def reset(self):
        cmd = "reset"
        env_info = self.barier.invoke(self.env, self.ind, cmd, self.cfg)

        state = env_info.vector_observations
        if -1 != self.ind:
            state = state[self.ind]

        return state

    def step(self, action):
        cmd = "step"
        env_info = self.barier.invoke(self.env, self.ind, cmd, action)

        state = env_info.vector_observations
        done = env_info.local_done
        reward = env_info.rewards

        if -1 != self.ind:
            state = state[self.ind]
            done = done[self.ind]
            reward = reward[self.ind]

        return (state, done, reward)

    def render(self):
        pass # well unity is not possible to turnon/off rendering ...

class UnityGroupSync:
    def __init__(self, num):
        self.lock = threading.RLock()
        self.pipes = [SimpleQueue() for _ in range(num + 1)]
        self.counter = { "reset" : [], "step" : [] }

    def _pass_trough(self, ind, cmd, data):
        if -1 == ind: # test policy
            return True

        with self.lock:
            self.counter[cmd] += [data]
            if len(self.counter[cmd]) + 1 != len(self.pipes):
                return False

        return True

    def _collect_data(self, ind, cmd, data):
        if -1 == ind:
            return data
        data = self.counter[cmd]
        self.counter[cmd] = []

        if "reset" in cmd:
            data = data[0]

        return data

    def _barier(self, env, ind, cmd, data):
        if not self._pass_trough(ind, cmd, data):
            return

        data = self._collect_data(ind, cmd, data)
        out = env.control(cmd, data)

        inds = range(len(self.pipes) - 1) if -1 != ind else [ind]
        for i in inds:
            self.pipes[i].put(out)

    def invoke(self, env, ind, cmd, data):
        self._barier(env, ind, cmd, data)
        data = self.pipes[ind].get()
        return data

def unity_factory(cfg, count):
    barier = UnityGroupSync(count)
    def factory(ind): # created remotely ~ at remote task manager process ( can be main proc )
        try:
            factory.env.ping()
        except:
            factory.env = UnityEnv(cfg)

        return UnityBrain(factory.env, barier, ind)
    return factory
