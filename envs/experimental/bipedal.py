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

from utils.task import Task
from utils.curiosity import *

class BipedalTask(Task):
    def __init__(self, cfg, encoder, xid = -1):
        self.env = gym.make(cfg['task'])
        self.encoder = encoder
        self.lock = threading.RLock()

        self.cfg = cfg
        init_state = self.reset()
        state_size = len(init_state)

        self.action_low = -1.
        self.action_high = 1.
        super(BipedalTask, self).__init__(
                cfg,
                xid,
                self.env,
                4,
                state_size)

        self.lock = threading.RLock()
        self.rewarder = CuriosityPrio(self, cfg)

    def reset(self, seed = None):
        state = super(BipedalTask, self).reset(seed)
        return state

    def new(self, i):
        if self.xid == -1:
            return BipedalTask(self.cfg, self.encoder, i)
        return super(BipedalTask, self).new(i)

    def step_ex(self, action, test = False):
        self.n_steps += 1

# call for action
        a = action
# interact with environment
        state, reward, done, _ = self.env.step(a)
# transform informations
        good = True#reward > 0#
        return action, state, reward, done, good

    def wrap_value(self, x):
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.tanh(x)

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards))
        return sum(rewards) > 300.

#    def normalize_state(self, states):
#        return self.encoder.normalize(np.array(states))

#    def update_normalizer(self, states):
#        with self.lock:
#            self.encoder.update(np.vstack(states))


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
            BipedalTask(cfg, encoder), # task "manager"
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
                    all(bot.task_main.test_policy(bot, True)[0] for _ in range(10)))
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
