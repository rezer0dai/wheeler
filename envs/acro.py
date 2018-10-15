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

ENV = gym.make(cfg['task'])

def transform(s):
    return np.hstack([
        np.arccos(s[0]),
        np.arccos(s[2]),
        s,
        ])

def wtf(s, n, gs, objective_id, her):
    return (-np.cos(s[1]) - np.cos(s[2]+s[1]), gs[0])

def fun_reward(s, n, gs, objective_id, her):
    return -1 * int(-np.cos(s[1]) - np.cos(s[2]+s[1]) <= gs[0])

def sample_goal(goal, target, n_target):
    def noise_goal():
        return [-np.cos(goal[1]) - np.cos(goal[2]+goal[1]) - 1e-3]
    gs = noise_goal()
    return (np.hstack([gs, target[1:]]).reshape(-1),
            np.hstack([gs, n_target[1:]]).reshape(-1))

# TODO : indexing in more nice way .. some abstraction -> wtf is 1 (state) 4 (next_state) 5 (n_reward) ...
def update_goal(her_target, goal, trajectory, objective_id, gamma):
    goal, n_goal = sample_goal(
            goal,
            her_target[0], her_target[4])

    rewards = map(
        lambda step: fun_reward(step[0], step[4], goal, objective_id, True), trajectory)
    reward = n_reward(rewards, gamma)

    assert cfg['n_step'] != 1 or reward == 0, "her malfunctioning {}".format(reward)

    return (goal, her_target[1], her_target[2], her_target[3], n_goal, [reward], her_target[6])

# TODO : move to utils .. TODO : create utils ..
def n_reward(rewards, gamma):
    return sum(map(lambda t_r: t_r[1] * gamma**t_r[0], enumerate(rewards)))

from utils.replay import *
from utils.tnorm import *

from agents.zer0bot import Zer0Bot

from utils.task import Task

class AcroBotTask(Task):
    def __init__(self, cfg, encoder, xid = -1):
        self.cfg = cfg
        self.env = gym.make(cfg['task'])
        self.encoder = encoder

        self.action_low = 0.
        self.action_high = 1.

        self.lock = threading.RLock()
        init_state = self.reset()
        state_size = len(init_state)

        super(AcroBotTask, self).__init__(
                cfg,
                xid,
                self.env,
                self.env.action_space.n,#3#
                state_size)

    def reset(self, seed = None, test = False):
        state = super().reset(seed)
        state = transform(state)
        self.prev_state = np.zeros(len(state))
        return state.reshape(-1)

    def new(self, i):
        if self.xid == -1:
            return AcroBotTask(self.cfg, self.encoder, i)
        return super(AcroBotTask, self).new(i)

    def step_ex(self, action, test = False):
        self.n_steps += 1

        action, a = self.softmax_policy(action, test)
        state, reward, done, _ = self.env.step(a)

        state = transform(state)

        out = '''
        while not reward and fun_reward(
#                np.hstack([[0], self.prev_state]),
                np.hstack([[0], state]),
                np.hstack([[0], state]),
                [1.], 0, False):
            self.env.render()
            print("inconsisten fun_rward with environment reward!", reward, done, wtf(np.hstack([[0], state]), np.hstack([[0], state]), [1.], 0, False), state)
        '''
        if done and not reward: print("\n","=" * 30," DONE WITH : ", self.n_steps, reward)
        self.prev_state = state

        return (action, state, reward, done, True)

    def wrap_value(self, x):
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.clamp(x, min=-1, max=+1)

    def goal_met(self, states, rewards, n_steps):
#        return sum(rewards) > -60.
        return sum(rewards) > -130

    def make_replay_buffer(self, cfg, actor):
        buffer_size = self.cfg['replay_size']
        return ReplayBuffer(cfg, self.xid, actor, update_goal)

    def her_state(self):
        return [1.]

    def normalize_state(self, states):
        states = np.array(states).reshape(-1, self.state_size() * self.cfg['history_count'] + self.cfg['her_state_size'])
        return self.encoder.normalize(states)

    def update_normalizer(self, states):
        states = np.vstack(states)
        with self.lock:
            self.encoder.update(states)

def main():
    print(cfg)

    import agents.ModelTorch as ModelTorch

    counter = 0
    while True:
        encoder = Normalizer(cfg['history_count'] * len(transform(ENV.reset())) + cfg['her_state_size'])

        counter += 1
        bot = Zer0Bot(
            cfg,
            AcroBotTask(cfg, encoder), # task "manager"
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
