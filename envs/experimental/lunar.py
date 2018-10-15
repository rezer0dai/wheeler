import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import time, threading, math, random, sys
from collections import deque, namedtuple, deque

import toml
cfg = toml.loads(open('cfg.toml').read())

torch.set_default_tensor_type(cfg['tensor'])

from agents.zer0bot import Zer0Bot
from utils.tnorm import *

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

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

from utils.task import Task
from utils.curiosity import *

class LunarTask(Task):
    def __init__(self, cfg, encoder, xid = -1):
        self.env = gym.make(cfg['task'])
        self.encoder = encoder
        self.lock = threading.RLock()

        self.cfg = cfg
        init_state = self.reset()
        state_size = len(init_state)

        self.action_low = 0.
        self.action_high = 1.
        super(LunarTask, self).__init__(
                cfg,
                xid,
                self.env,
                4 if 'cont' not in cfg['task'].lower() else 2,
                state_size)

        self.lock = threading.RLock()
        self.rewarder = CuriosityPrio(self, cfg)

    def reset(self, seed = None, test = False):
        state = super(LunarTask, self).reset(seed, test)
#        state = self.encoder.transform(state)
        return state

    def new(self, i):
        if self.xid == -1:
            return LunarTask(self.cfg, self.encoder, i)
        return super(LunarTask, self).new(i)

    def step_ex(self, action, test = False):
        self.n_steps += 1

# call for action
        if 'cont' not in self.cfg['task'].lower():
            action, a = self.softmax_policy(action, test)
        else:
            a = action
# interact with environment
        state, reward, done, _ = self.env.step(a)
        if not self.xid: self.env.render()
#        if not test: reward = reward if np.abs(reward) < 7 else np.sign(reward)
# transform informations
#        good = True#reward > 0#
        good = True if 0 == self.xid else ((sum(state[-2:]) > 0) or done)
#        if not test and 1 == self.xid: reward = reward - 1
#        if not test and 0 == self.xid: reward = max(reward, np.sign(reward))

#        state = self.encoder.transform(state)
        return action, state, reward, done, good

    def wrap_value(self, x):
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.tanh(x)
        return torch.clamp(x, min=-1., max=+1.)

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards))
        return sum(rewards) > 200.

    def normalize_state(self, states):
        return self.encoder.normalize(np.array(states))

    def update_normalizer(self, states):
        with self.lock:
            self.encoder.update(np.vstack(states))


import agents.ModelTorch as ModelTorch

def main():
    global cfg
    print(cfg)
    counter = 0
    while True:
#        encoder = State(gym.make(cfg['task']))
        encoder = Normalizer(len(gym.make(cfg['task']).reset()) * cfg['history_count'])

        counter += 1
        bot = Zer0Bot(
            cfg,
            LunarTask(cfg, encoder), # task "manager"
            ModelTorch.ActorNetwork,
            ModelTorch.CriticNetwork)

        bot.start()

        z = 0
        ROUNDS = cfg['mcts_rounds']
        bot.task_main.training_status(False)
        while not bot.task_main.learned():
            bot.train(ROUNDS)
            print()
            bot.task_main.training_status(
                    all(bot.task_main.test_policy(bot, False)[0] for _ in range(10)))
            z+=1

        print("\n")
        print("="*80)
        print("training over", counter, z * bot.task_main.subtasks_count() * ROUNDS)
        print("="*80)
        for _ in range(10): print("total steps : ", len(bot.task_main.test_policy(bot, False)[2]))
        while True: bot.task_main.test_policy(bot, True)
        break

if '__main__' == __name__:
    main()
