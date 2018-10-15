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

from utils.task import Task

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

class CarTTask(Task):
    def __init__(self, cfg, xid = -1):
        self.cfg = cfg
        self.env = gym.make(cfg['task'])

        self.action_low = 0.
        self.action_high = 1.

        super(CarTTask, self).__init__(cfg, xid, self.env, 2, len(self.env.reset()))

        self.reset()

# default
        self.DONE = 1.
        self.FAIL = -40.
        self.CONT = 1.

        if 0 == xid:
            print("XID0")
            self.DONE = 1.
            self.FAIL = -40.
            self.CONT = 1.
        if 1 == xid:
            print("XID1")
            self.DONE = .1
            self.FAIL = -4.
            self.CONT = .1
        if 2 == xid:
            print("XID2")
            self.DONE = 40.
            self.FAIL = -10.
            self.CONT = .1

    def reset(self, seed = None, test = False):
        state = super(CarTTask, self).reset(seed, test)
        return state

    def new(self, i):
        if self.xid == -1:
            return CarTTask(self.cfg, i)
        return super(CarTTask, self).new(i)

    def step_ex(self, action, test = False):
        self.n_steps += 1

        action, a = self.softmax_policy(action, test)

        state, reward, done, _ = self.env.step(a)
#        state = self.state.transform(state)
#        return action, state, reward, done, True

        if done:
            if self.n_steps > self.max_n_episode():
                return action, state, self.DONE, True, False
            else:
                return action, state, self.FAIL, True, True

        return action, state, self.CONT, False, True

    def wrap_value(self, x):
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
#        return torch.tanh(x)
#        return F.sigmoid(x)
        return torch.clamp(x, min=-1, max=+2)

    def goal_met(self, states, rewards, n_steps):
        return len(rewards) > 199


def main():
    print(cfg)

    import agents.ModelTorch as ModelTorch

    counter = 0
    while True:
        counter += 1
        bot = Zer0Bot(
            cfg,
            CarTTask(cfg), # task "manager"
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
        for i in range(10): print("total steps : ", len(bot.task_main.test_policy(bot, True)[2]))
        while True: bot.task_main.test_policy(bot, True)
        break

if '__main__' == __name__:
    main()
