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

from utils.rbf import *

from utils.task import Task
from utils.curiosity import *

class MCarTask(Task):
    def __init__(self, cfg, encoder, xid = -1):
        self.env = gym.make(cfg['task'])
        self.encoder = encoder

        self.cfg = cfg

        self.reward = 0
        self.rewards = []
        init_state = self.reset()
        state_size = len(init_state)

        self.action_low = -1.
        self.action_high = +1.
        super(MCarTask, self).__init__(
                cfg,
                xid,
                self.env,
                1,
                state_size)

        self.rewarder = CuriosityPrio(self, cfg)

    def reset(self, seed = None, test = False):
        state = super(MCarTask, self).reset(seed, test)
        self.rewards.append(self.reward)
        self.reward = 0
        self.prev_state = state
        return self.encoder.transform(state)

    def new(self, i):
        if self.xid == -1:
            return MCarTask(self.cfg, self.encoder, i)
        return super(MCarTask, self).new(i)

    def step_ex(self, action, test = False):
        self.n_steps += 1
        state, reward, done, _ = self.env.step(action)

        self.reward += (done and reward > 0)

#        if not test:
#            s = self.encoder.transform(self.prev_state)
#            n = self.encoder.transform(state)
#            ss, ns = np.vstack([s] * self.cfg['history_count']), np.vstack([n] * self.cfg['history_count'])
#            ss = ss.reshape(1, -1)
#            ns = ns.reshape(1, -1)
#            reward = self.rewarder.weight(ss, ns, action)[0]
#            self.rewarder.update(ss, ns, action)
        self.prev_state = state

        if not test and sum(self.rewards) < 3 and not done:
            true_state = np.abs(np.cos(np.pi/3.) + state[0])
            reward += -(1. - true_state)

        return action, self.encoder.transform(state), reward, done, True

    def wrap_value(self, x):
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.tanh(x)
        return torch.clamp(x, min=-1, max=+2)

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards))
        return sum(rewards) > 90.

def main():
    print(cfg)

    import agents.ModelTorch as ModelTorch
    encoder = State(gym.make(cfg['task']), [5., 2., 1., .5], [20] * 4)
    task = MCarTask(cfg, encoder)
    counter = 0
    while True:
        counter += 1
        bot = Zer0Bot(
            0, cfg,
            task, # task "manager"
            ModelTorch.ActorNetwork,
            ModelTorch.CriticNetwork)

        bot.start()

        z = 0
        task.training_status(False)
        while not task.learned():
            bot.train()
            print()
            task.training_status(
                    all(task.test_policy(bot, True)[0] for _ in range(10)))
            z+=1

        print("\n")
        print("="*80)
        print("training over", counter, z * bot.task_main.subtasks_count() * cfg['mcts_rounds'])
        print("="*80)

        for i in range(10): print("total steps : %i < training : %i :: %i >"%(
            counter, 
            z * cfg['mcts_rounds'] * task.subtasks_count(), 
            len(task.test_policy(bot, True)[2])))

        while True: bot.task_main.test_policy(bot, True)
        break

if '__main__' == __name__:
    main()
