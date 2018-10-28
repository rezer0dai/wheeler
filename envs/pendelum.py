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

class PendelumTask(Task):
    def __init__(self, cfg, xid = -1):
        # utils
        self.cfg = cfg
        self.env = gym.make(cfg['task'])

        init_state = self.reset()
        state_size = len(init_state)

        # helpers ~ TODO minirefactor, move it to params of Task.__init__!
        self.action_low = self.env.action_space.low[0]
        self.action_high = self.env.action_space.high[0]
        super(PendelumTask, self).__init__(
                cfg,
                xid,
                self.env,
                1,#self.env.action_space.n,
                state_size)

    def new(self, i):
        if self.xid == -1:
            return PendelumTask(self.cfg, i)
        return super(PendelumTask, self).new(i)

    def step_ex(self, action, test = False):
        self.n_steps += 1
#        action = F.tanh(torch.from_numpy(action)) * 2
        state, reward, done, _ = self.env.step(action)
        return (action, state, reward, done, True)

    def wrap_value(self, x):
#        return F.tanh(x / 2.) * self.cfg['max_reward_val']
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.clamp(x, min=-2, max=+2)
        return F.tanh(x) * 2

    def goal_met(self, states, rewards, n_steps):
        print("TEST ", sum(rewards))
        return sum(rewards) > -150.

def main():
    print(cfg)

    counter = 0
    while True:
        import agents.ModelTorch as ModelTorch

        counter += 1
        bot = Zer0Bot(
            0,
            cfg,
            PendelumTask(cfg), # task "manager"
            ModelTorch.ActorNetwork,
            ModelTorch.CriticNetwork)

        bot.start()

        z = 0
        bot.task_main.training_status(False)
        while not bot.task_main.learned():
            bot.train()
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
