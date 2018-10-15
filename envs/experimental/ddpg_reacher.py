import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

from unityagents import UnityEnvironment
import numpy as np

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
cfg['her_max_ratio'] = 0
cfg['her_state_size'] = 0
cfg['her_state_features'] = 0

torch.set_default_tensor_type(cfg['tensor'])

from agents.zer0bot import Zer0Bot
from utils.task import Task
from utils.curiosity import *
from utils.replay import *
from utils.tnorm import *

CLOSE_ENOUGH = 1.25#1.2#5.0#.05

class FetchNReachTask(Task):
    env = UnityEnvironment(file_name="/home/xxai/unity/Reacher_Linux/Reacher.x86_64")
    def __init__(self, cfg, xid = -1):
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        self.lock = threading.RLock()

        self.cfg = cfg
        self.states = []
        init_state = self.reset()
        state_size = len(init_state)

        self.action_low = -1.
        self.action_high = 1.
        super(FetchNReachTask, self).__init__(
                cfg,
                xid,
                self.env,
                4,
                state_size)

    def env_reset(self, _, test):
        print("STATIC" if not test else "MOVING")
        env_info = self.env.reset(config={"goal_size":CLOSE_ENOUGH * 4, "goal_speed":.1}, train_mode=True)[self.brain_name]
#        env_info = self.env.reset(config={"goal_size":CLOSE_ENOUGH * 4, "goal_speed":1.0 if test else random.randint(0, 8) / 4}, train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]
        return state

    def reset(self, seed = None, test = False):
        state = super().reset(seed, test)
        return state.reshape(-1)

    def new(self, i):
        if self.xid == -1:
            return FetchNReachTask(self.cfg, i)
        return super(FetchNReachTask, self).new(i)

    def make_replay_buffer(self, cfg, actor):
        buffer_size = self.cfg['replay_size']
        return ReplayBuffer(cfg, self.xid, actor)

    def step_ex(self, action, test = False):
        self.n_steps += 1

        env_info = self.env.step(action)[self.brain_name]
        state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        reward = env_info.rewards[0]
        good = True
        return action, state, reward, done, good

    def wrap_value(self, x):
#        return torch.tanh(x) * self.cfg['max_reward_val']#1000.#
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.clamp(x, min=-1, max=+1)
        return torch.tanh(x)
        return (torch.sigmoid(x) - .5) * 2

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards), sum(map(lambda r: r != 0, rewards)), len(rewards))
        return sum(abs(r) for r in rewards) > 30

import agents.ModelTorch as ModelTorch

def main():
    global cfg
    print(cfg)
    counter = 0
    while True:
        counter += 1
        bot = Zer0Bot(
            cfg,
            FetchNReachTask(cfg), # task "manager"
            ModelTorch.ActorNetwork,
            ModelTorch.CriticNetwork)

        bot.start()

        z = 0
        ROUNDS = cfg['mcts_rounds']
        bot.task_main.training_status(False)

#        while True: bot.task_main.test_policy(bot, False)

        while not bot.task_main.learned():
            bot.train(ROUNDS)
            print()
            bot.task_main.training_status(
                    sum(bot.task_main.test_policy(bot, False)[0] for _ in range(3)) == 3)
            z+=1

        print("\n")
        print("="*80)
        print("training over", counter, z * bot.task_main.subtasks_count() * ROUNDS)
        print("="*80)
        for _ in range(10): print("total reward : ", sum(bot.task_main.test_policy(bot, False)[2]))
        while True: bot.task_main.test_policy(bot, False)
        break

if '__main__' == __name__:
    main()

#'''

