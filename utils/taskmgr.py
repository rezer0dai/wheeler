from torch.multiprocessing import Queue, SimpleQueue, Process

class RemoteTaskComm(threading.Thread):
    def __init__(self, factory, pipe_cmd, pipe_data):
        super().__init__()
        self.pipe_cmd = pipe_cmd
        self.pipe_data = pipe_data

        self.factory = factory

        self.cmd = { 
                "reset" : self._reset,
                "step" : self._step, }

    def run(self): # active only at main process
        self.mgr = LocalTaskManager(self.factory)

        while True:
            data = self.pipe_cmd.get()
            if not data:
                break
            cmd, data = data
            if cmd not in self.cmd:
                break
            ind, out = self.cmd[cmd](*data)
            self.pipe_data[ind].put(out)

    def _reset(self, ind, key, seed):
        return ind, self.mgr.reset(key, seed)

    def _step(self, ind, key, action):
        return ind, self.mgr.step(key, action)

# all bots needs to be created from one process, and all task must be created in their ctors!!
# > aka fully initialized before multiprocessing and training will start!
# > haha what a design conditions .. but its ok for this mini project
class RemoteTaskManager:
    def __init__(self, factory_task, factory_env, n_tasks):
        self.pipe_cmd = Queue()
        self.pipe_data = [Queue() for _ in range(n_tasks)]

        self.factory_task = factory_task

# create process which will handle requests!
        self.com = RemoteTaskComm(factory_env, self.pipe_cmd, self.pipe_data)

        self.dtb = {}
        self.lock = threading.RLock()
#    def turnon():
        self.com.start()

    def _ind(self, bot_id, objective_id):
        key = (bot_id, objective_id)
        with self.lock:
            return self.dtb[key]

    def new(self, bot_id, objective_id):
        key = (bot_id, objective_id)
        with self.lock: # multibot register
            if key not in self.dtb:
                self.dtb[key] = len(self.dtb)
            assert len(self.dtb) <= len(self.pipe_data), "tasks more than pipe.. remote task manager problem .."
        return self.factory_task(bot_id, objective_id, self) # will wait until each reset

    def reset(self, bot_id, objective_id, seed):
        ind = self._ind(bot_id, objective_id)
        args = (ind, (bot_id, objective_id), seed)
        self.pipe_cmd.put(["reset", args])
        return self.pipe_data[ind].get()

    def step(self, bot_id, objective_id, action, test):
        ind = self._ind(bot_id, objective_id)
        args = (ind, (bot_id, objective_id), action)
        self.pipe_cmd.put(["step", args])
        return self.pipe_data[ind].get()

class LocalTaskManager:
    def __init__(self, factory):
        self.tasks = {}
        self.factory = factory
        self.lock = threading.RLock()

    def _task(self, key):
        with self.lock:
            if key not in self.tasks:
                self.tasks[key] = self.factory(len(self.tasks))
            return self.tasks[key]

    def reset(self, key, seed):
        task = self._task(key)
        task.seed(seed)
        return task.reset()

    def step(self, key, action):
        task = self._task(key)
        if key[0] == -1:
            task.render()
        return task.step(action)

