def unity_factory(ind):
    return UnityBrain(ind)

class UnityBrain:
#    env = UnityEnvironment(file_name="/home/xxai/unity/Reacher_Linux/Reacher.x86_64")
    def __init__(self, ind):
        self.ind = ind
        self.brain_name = self.env.brain_name[self.ind]
        self.cfg = None

    def seed(cfg):
        self.cfg = cfg

    def step(action):
        return self.env.step(action)[self.brain_name]

    def reset():
        return self.env.reset(config=cfg, train_mode=True)[self.brain_name]


