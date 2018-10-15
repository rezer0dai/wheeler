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

torch.set_default_tensor_type(cfg['tensor'])

from agents.zer0bot import Zer0Bot
from utils.task import Task
from utils.curiosity import *
from utils.replay import *

from utils.tnorm import *

CLOSE_ENOUGH = 1.25#1.2#5.0#.05

def transform(obs):
    return np.hstack([
        obs[3+4+3+3:3+4+3+3+3],
        obs[:3+4+3+3],
        obs[3+4+3+3+3:-4-3],
#        obs[-4:],
        ])


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def fun_reward(s, n, gs, objective_id, her):
#    print("how close ", np.abs(goal_distance(n[3:3+3], gs[:3])))
#    print(".. ", n[3:3+3], gs[:3])
    hs = cfg['her_state_size']
    ind = -(len(n) - hs) // cfg['history_count']
    return -1 * (2 * CLOSE_ENOUGH < np.abs(goal_distance(s[ind:ind+hs], gs[:hs])))
    xid = objective_id % 4
    if xid < 3:
        a = np.abs(n[3+xid] - gs[xid])
        b = np.abs(s[3+xid] - gs[xid])
    else:
        a = np.abs(goal_distance(n[3:3+3], gs[:3]))
        b = np.abs(goal_distance(s[3:3+3], gs[:3]))
    if b < CLOSE_ENOUGH:
        return 0.#3.
#    if her: print("HER", a, b)
    return -1 + .9 * int(a < b)

def sample_goal(goal, target, n_target):
    def noise_goal():
        hs = cfg['her_state_size']
        ind = -(len(goal) - hs) // cfg['history_count']
#        return goal[ind:ind+hs]
        for i in range(1):# be carefull extremly expensive
            radius = np.abs(np.random.rand() * CLOSE_ENOUGH)
            angle = np.random.rand() * np.pi * 2
            a = np.cos(angle) * radius
            b = np.sin(angle) * radius
            ids = np.random.choice(hs, 2, p=[1/hs]*hs, replace=False)

            g = goal.copy()
            g[ids[0]] = g[ids[0] + ind] + a
            g[ids[1]] = g[ids[1] + ind] + b

            if np.abs(goal_distance(g[:hs], g[ind:ind+hs])) < CLOSE_ENOUGH:
                return g[:hs]
        return goal[ind:ind+hs]
    gs = noise_goal()
    return (np.hstack([gs, target[3:]]).reshape(-1),
            np.hstack([gs, n_target[3:]]).reshape(-1))

# TODO : indexing in more nice way .. some abstraction -> wtf is 1 (state) 4 (next_state) 5 (n_reward) ...
def update_goal(her_target, goal, trajectory, objective_id, gamma):
    goal, n_goal = sample_goal(
            goal,
            her_target[0], her_target[4])

    rewards = map(
            lambda step: fun_reward(step[0], step[4], goal, objective_id, True), trajectory)
    reward = n_reward(rewards, gamma)

    return (goal, her_target[1], her_target[2], her_target[3], n_goal, [reward], her_target[6])

# TODO : move to utils .. TODO : create utils ..
def n_reward(rewards, gamma):
    return sum(map(lambda t_r: t_r[1] * gamma**t_r[0], enumerate(rewards)))

class FetchNReachTask(Task):
    env = UnityEnvironment(file_name="/home/xxai/unity/Reacher_Linux/Reacher.x86_64")
    def __init__(self, cfg, encoder, xid = -1):
#        self.env = gym.make(cfg['task'])

        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        self.encoder = encoder
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

#        self.rewarder = CuriosityPrio(self, cfg)

    def env_reset(self, _, test):
        print("STATIC" if not test else "MOVING")
        env_info = self.env.reset(config={"goal_size":CLOSE_ENOUGH * 4, "goal_speed":.1}, train_mode=True)[self.brain_name]
#        env_info = self.env.reset(config={"goal_size":CLOSE_ENOUGH * 4, "goal_speed":1.0 if test else random.randint(0, 8) / 4}, train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]
        return state

    def reset(self, seed = None, test = False):
        state = super().reset(seed, test)

        self.goal = state[-4-3:-1-3]
        state = transform(state)
        self.prev_state = np.zeros(len(state))
        return state.reshape(-1)

    def update_normalizer(self, states):
        states = np.vstack(states)
        with self.lock:
#            self.encoder.update(states)
            self.encoder[0].update(states[:, 3:])
            self.encoder[1].update(states[:, :3])
            self.encoder[1].update(states[:, 3:6])
        return

    def new(self, i):
        if self.xid == -1:
            return FetchNReachTask(self.cfg, self.encoder, i)
        return super(FetchNReachTask, self).new(i)

    def make_replay_buffer(self, cfg, actor):
        buffer_size = self.cfg['replay_size']
        return ReplayBuffer(cfg, self.xid, actor, update_goal)

    def step_ex(self, action, test = False):
        self.n_steps += 1

        env_info = self.env.step(action)[self.brain_name]
        state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        r = env_info.rewards[0]

        self.goal = state[-4-3:-1-3]
        state = transform(state)
        good = True

#        reward = r if test else -(r == 0)

        reward = r if test else fun_reward(np.hstack(
            [self.goal, np.hstack([self.prev_state] * cfg['history_count'])]),
                np.hstack([self.goal, np.hstack([state] * cfg['history_count'])]),
                self.goal, self.xid, False)

#        reward = r if test else fun_reward(np.hstack([self.goal, self.prev_state]),
#                np.hstack([self.goal, state]), self.goal, self.xid, False)
        self.prev_state = state

#        if r != 0: print("OK WELL DONE ", self.n_steps, reward)
#        n = np.hstack([self.goal, state])
#        gs = np.hstack([self.goal, state])
#        print(r, reward)
#        assert not r or reward == 0, "incosistent rewards with goal {}::{} || {} -> {} {}".format(r, reward,
#                np.abs(goal_distance(n[3:3+3], gs[:3])), state, self.goal)

        return action, state, reward, done, good

    def wrap_value(self, x):
#        return torch.tanh(x) * self.cfg['max_reward_val']#1000.#
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return (torch.sigmoid(x) - .5) * 2
        return torch.tanh(x)
        return torch.clamp(x, min=-1, max=+1)

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards), sum(map(lambda r: r != 0, rewards)), len(rewards))
        return sum(abs(r) for r in rewards) > 30

    def her_state(self):
        return self.goal

    def normalize_state(self, states): # here is ok to race ~ well not ok but i dont care now :)
        states = np.array(states).reshape(-1, self.state_size() * self.cfg['history_count'] + self.cfg['her_state_size'])
        s = self.encoder[0].normalize(states[:, self.cfg['her_state_size']:])
        g = self.encoder[1].normalize(states[:, :self.cfg['her_state_size']])
        return np.hstack([s, g])

import agents.ModelTorch as ModelTorch

def main():
    global cfg
    print(cfg)
    counter = 0
    while True:
        encoder_s = Normalizer((30-4) * cfg['history_count'])
        encoder_g = Normalizer(cfg['her_state_size'])

        counter += 1
        bot = Zer0Bot(
            cfg,
            FetchNReachTask(cfg, (encoder_s, encoder_g)), # task "manager"
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
