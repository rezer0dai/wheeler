import threading, abc
from torch.multiprocessing import SimpleQueue, Process, Queue

class ITaskWorker:
    @abc.abstractmethod
    def register(self, bot_id, objective_id, gl):
        pass
    @abc.abstractmethod
    def reset(self, bot_id, objective_id, seed):
        pass
    @abc.abstractmethod
    def step(self, bot_id, objective_id, action):
        pass

# decide if we want to push it to separate process or in main proc as thread
class RemoteTaskServer(Process):#threading.Thread):#
    def __init__(self, factory_mgr, factory_env, pipe_cmd, pipe_data):
        super().__init__()
        self.pipe_cmd = pipe_cmd
        self.pipe_data = pipe_data
        self.workers = [None] * len(pipe_data)

        self.factory_mgr = factory_mgr
        self.factory_env = factory_env

        self.cmd = {
                "reset" : self._reset,
                "step" : self._step, }

    def run(self): # active only at main process
        self.mgr = self.factory_mgr(self.factory_env)

        while True:
            data = self.pipe_cmd.get()
            if not data:
                break
            cmd, data = data
            if cmd not in self.cmd:
                break

            ind, key, info = data

            if self.workers[ind] is not None:
                self.workers[ind].join()

            self.workers[ind] = threading.Thread(
                    target=self.async_processing,
                    args=(ind, cmd, key, info))
            self.workers[ind].daemon = True

            self.workers[ind].start()

    def async_processing(self, ind, cmd, key, data):
        ind, out = self.cmd[cmd](ind, key, data)
        self.pipe_data[ind].put(out)

    def _reset(self, ind, key, seed):
        return ind, self.mgr.reset(*key, seed)

    def _step(self, ind, key, action):
        return ind, self.mgr.step(*key, action)

# all bots needs to be created from one process, and all task must be created in their ctors!!
# > aka fully initialized before multiprocessing and training will start!
# > haha what a design conditions .. but its ok for this mini project
class RemoteTaskManager:
    def __init__(self, factory_env, factory_mgr, n_tasks):
        self.pipe_cmd = Queue() # we want to queue more data in a row
        self.pipe_data = [SimpleQueue() for _ in range(n_tasks + 1)]

        self.factory_mgr = factory_mgr

# create thread ( in main process!! ) which will handle requests!
        self.com = RemoteTaskServer(factory_mgr, factory_env, self.pipe_cmd, self.pipe_data)

        self.dtb = {}
        self.lock = threading.RLock()

#    def turnon():
        self.com.start()

    def _ind(self, bot_id, objective_id):
        key = (bot_id, objective_id)
        assert key in self.dtb, "you forgot to register your environment [remote] in ctor!!"
        with self.lock:
            return self.dtb[key]

    def register(self, bot_id, objective_id):
        key = (bot_id, objective_id)
        with self.lock: # multibot register
            assert key not in self.dtb, "double initialization of your environment [remote]!!"
            self.dtb[key] = len(self.dtb)
            assert len(self.dtb) <= len(self.pipe_data), "#tasks > #pipes [remote task manager]"

    def reset(self, bot_id, objective_id, seed):
        ind = self._ind(bot_id, objective_id)
        args = (ind, (bot_id, objective_id), seed)
        self.pipe_cmd.put(["reset", args])
        return self.pipe_data[ind].get()

    def step(self, bot_id, objective_id, action):
        ind = self._ind(bot_id, objective_id)
        args = (ind, (bot_id, objective_id), action)
        self.pipe_cmd.put(["step", args])
        return self.pipe_data[ind].get()

class LocalTaskManager:
    def __init__(self, factory):
        self.envs = {}
        self.factory = factory
        self.lock = threading.RLock()

    def _env(self, key):
        if key not in self.envs:
            self.register(*key) # remote branch ...
        return self.envs[key]

    def register(self, bot_id, objective_id):
        key = (bot_id, objective_id)
        with self.lock:
            assert key not in self.envs, "double initialization of your environment!!"
            self.envs[key] = self.factory(len(self.envs))

    def reset(self, bot_id, objective_id, seed):
        env = self._env((bot_id, objective_id))
        env.seed(seed)
        return env.reset()

    def step(self, bot_id, objective_id, action):
        env = self._env((bot_id, objective_id))

        if -1 == objective_id: # we are testing our policy!
            env.render()

        return env.step(action)
