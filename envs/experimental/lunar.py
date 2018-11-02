import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import numpy as np
import toml, torch, gym

from utils.task import Task
from utils.taskinfo import *
from utils.taskmgr import *

from utils.normalizer import *

from agents.zer0bot import Zer0Bot
import agents.ModelTorch as ModelTorch

from utils.curiosity import *
from utils.replay import *

class LunarLanderTask(Task):
    def __init__(self, 
            cfg, 
            env, 
            objective_id, bot_id, 
            action_low, action_high, state_size,
            encoder):

        super().__init__(
                cfg,
                env,
                objective_id,
                bot_id,
                action_low, action_high)

        self.encoder = encoder
        self.state_size = state_size

    def step_ex(self, action, test = False):
        if len(action) == 4:
            action, a = self.softmax_policy(action, test)
        else:
            a = action

        state, reward, done, _ = self.env.step(self.bot_id, self.objective_id, a)
        good = True if 1 == self.objective_id else ((sum(state[-2:]) > 0) or done)
        if test: return action, [ state.reshape(-1) ], reward, done, True
        return action, state, reward, done, True

    def goal_met(self, states, rewards, n_steps):
        print("TEST ", sum(rewards))
        return sum(rewards) > 200

    def normalize_state(self, states):
        states = np.array(states).reshape(
                -1, 
                self.state_size * self.cfg['history_count'] + self.cfg['her_state_size'])
        return self.encoder.normalize(states)

    def update_normalizer(self, states):
        states = np.vstack(states)
        self.encoder.update(states)

class LunarLanderInfo(TaskInfo):
    def __init__(self, cfg, encoder, replaybuf, factory, Mgr, args):
        env = self.factory(0)
        super().__init__(
                len(env.reset()), 4 if 'cont' not in cfg['task'].lower() else 2,
                0, 1,
                cfg,
                encoder, replaybuf,
                factory, Mgr, args)


    def new(self, cfg, bot_id, objective_id):
        return LunarLanderTask(cfg, 
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high, self.state_size,
                self.encoder)

    @staticmethod
    def factory(ind): # bare metal task creation
        print("created %i-th task"%ind)
        CFG = toml.loads(open('cfg.toml').read())
        return gym.make(CFG['task'])

def main():
    CFG = toml.loads(open('cfg.toml').read())
    torch.set_default_tensor_type(CFG['tensor'])

    print(CFG)

    DDPG_CFG = toml.loads(open('gym.toml').read())

    ENV = gym.make(CFG['task'])
    state_size = len(ENV.reset())
    encoder = Normalizer(DDPG_CFG['history_count'] * state_size)

    INFO = LunarLanderInfo(
            CFG, 
            encoder, 
            ReplayBuffer, 
            LunarLanderInfo.factory, LocalTaskManager, ())

    task = INFO.new(DDPG_CFG, 0, -1)

    bot = Zer0Bot(
        0,
        DDPG_CFG,
        INFO,
        ModelTorch.ActorNetwork,
        ModelTorch.CriticNetwork)

    bot.turnon()

    z = 0
    task.training_status(False)
    while not task.learned():
        bot.train()
        print()
        task.training_status(
                all(task.test_policy(bot)[0] for _ in range(10)))
        z+=1

    print("\n")
    print("="*80)
    print("training over", z * CFG['n_simulations'] * CFG['mcts_rounds'])
    print("="*80)

    for i in range(10): print("total steps : training : %i :: %i >"%(
        z * CFG['mcts_rounds'] * CFG['n_simulations'],
        len(task.test_policy(bot)[2])))

if '__main__' == __name__:
    main()

lunar_rbf_encoding_poc = '''

class State:
    def __init__(self, env):
#        return
        def lunar_sample():
            FPS    = 50
            SCALE  = 30.0
            LEG_DOWN = 18
            VIEWPORT_W = 600
            VIEWPORT_H = 400

            HELIPAD_H = VIEWPORT_H/SCALE/4
            HELIPAD_B = HELIPAD_H-HELIPAD_H/2
            HELIPAD_T = HELIPAD_H+HELIPAD_H/2

            x = np.random.rand() * VIEWPORT_W
            y = HELIPAD_H/2 + abs(np.random.rand()) * (VIEWPORT_H - HELIPAD_H/2)
            if 0 == random.randint(0, 3):
                y = HELIPAD_H + abs(np.random.rand()) * (VIEWPORT_H/SCALE/2 - HELIPAD_H)

            return [
                (x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
                (y - VIEWPORT_H/SCALE/2) / (VIEWPORT_W/SCALE/2),
                np.random.rand()*2*np.pi*(VIEWPORT_W/SCALE/2)/FPS,
                np.random.rand()*2*np.pi*(VIEWPORT_H/SCALE/2)/FPS,
                np.random.rand()*2*np.pi,
                20.0*(np.random.rand())*2*np.pi/FPS,
                1.0 if y > HELIPAD_B and y < HELIPAD_T and random.randint(0, 3) else 0.0,
                1.0 if y > HELIPAD_B and y < HELIPAD_T and random.randint(0, 3) else 0.0
                ]
        observation_examples = np.array([lunar_sample() for x in range(1000000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        self.featurizer = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=64)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=64)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=128)),
                ("rbfy", RBFSampler(gamma=1.5, n_components=128)),
                #  ("rbfx", RBFSampler(gamma=0.1, n_components=33)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=128))
                ])
        self.featurizer.fit(self.scaler.transform(observation_examples))
        print("RBF-sampling done!")

    def transform(self, state):
 #       return state#
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
'''
