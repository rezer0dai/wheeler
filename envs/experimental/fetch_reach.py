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
from utils.curiosity import *
from utils.replay import *

from utils.tnorm import *

CLOSE_ENOUGH = .05

def transform(obs):
    return np.hstack([
        obs["achieved_goal"],
        obs["observation"][3:],
        ])

ENV = gym.make("FetchReach-v1")

def fun_reward(s, n, gs, objective_id, her):
    return ENV.compute_reward(s[3:6], gs[:3], None)
    xid = objective_id % 4
    if xid < 3:
        a = np.abs(n[3+xid] - gs[xid])
        b = np.abs(s[3+xid] - gs[xid])
    else:
        a = np.abs(gym.envs.robotics.fetch_env.goal_distance(n[3:3+3], gs[:3]))
        b = np.abs(gym.envs.robotics.fetch_env.goal_distance(s[3:3+3], gs[:3]))
    if b < CLOSE_ENOUGH:
        return 0.#3.
#    if her: print("HER", a, b)
    return -1 + .9 * int(a < b)

def sample_goal(goal, target, n_target):
    def noise_goal():
#        return goal[3:6]
        for i in range(3):# be carefull extremly expensive
            radius = np.abs(np.random.rand() * CLOSE_ENOUGH)
            angle = np.random.rand() * np.pi * 2
            a = np.cos(angle) * radius
            b = np.sin(angle) * radius
            ids = np.random.choice(3, 2, p=[1/3]*3, replace=False)

            g = goal.copy()
            g[ids[0]] = g[ids[0] + 3] + a
            g[ids[1]] = g[ids[1] + 3] + b

            if np.abs(gym.envs.robotics.fetch_env.goal_distance(g[:3], g[3:2*3])) < CLOSE_ENOUGH:
                return g[:3]
        return goal[3:6]
    gs = noise_goal()
    return (np.hstack([gs, target[3:]]).reshape(-1),
            np.hstack([gs, n_target[3:]]).reshape(-1))

# TODO : indexing in more nice way .. some abstraction -> wtf is 1 (state) 4 (next_state) 5 (n_reward) ...
def update_goal(her_target, trajectory, objective_id, gamma):
    goal, n_goal = sample_goal(
#            trajectory[random.randint(0, len(trajectory)-1)][0],
            trajectory[random.randint(0, len(trajectory)-1)][0],
            her_target[0], her_target[4])

    rewards = map(
            lambda step: fun_reward(step[0], step[4], goal, objective_id, True), trajectory)
    reward = n_reward(rewards, gamma)

    return (goal, her_target[1], her_target[2], her_target[3], n_goal, [reward], her_target[6])

# TODO : move to utils .. TODO : create utils ..
def n_reward(rewards, gamma):
    return sum(map(lambda t_r: t_r[1] * gamma**t_r[0], enumerate(rewards)))

class FetchNReachTask(Task):
    def __init__(self, cfg, encoder, xid = -1):
        self.env = gym.make(cfg['task'])
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

    def reset(self, seed = None):
        state = super().reset(seed)
        self.goal = state["desired_goal"]
        state = transform(state)
        self.prev_state = np.zeros(len(state))
        return state.reshape(-1)

    def update_normalizer(self, states):
        states = np.vstack(states)
        with self.lock:
            self.encoder.update(states)
#            self.encoder[0].update(states[:, 3:])
#            self.encoder[1].update(states[:, :3])
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

        state, reward, done, _ = self.env.step(action)

        self.goal = state["desired_goal"]
        state = transform(state)
        good = True

#        reward = fun_reward(np.hstack([self.goal, self.prev_state]),
#                np.hstack([self.goal, state]), self.goal, self.xid, False)
#        self.prev_state = state

#        done = done if done else (-0 == reward)

#        if (-0 == reward) and self.n_steps > 2: print("WE ARE DONE !!!", self.n_steps)

#        if not self.xid: self.env.render()
        return action, state, reward, done, good

    def wrap_value(self, x):
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.tanh(x)

    def goal_met(self, states, rewards, n_steps):
#        print("TEST : ", sum(rewards), len(rewards))
        return any(-0 == r for r in rewards)
        return -5 < sum(rewards[len(rewards)//2:])

    def her_state(self):
        return self.goal

    def normalize_state(self, states): # here is ok to race ~ well not ok but i dont care now :)
        states = np.array(states).reshape(-1, self.state_size() * self.cfg['history_count'] + self.cfg['her_state_size'])
        return self.encoder.normalize(states)
        s = self.encoder[0].normalize(states[:, 3:])
        g = self.encoder[1].normalize(states[:, :3])
        return np.hstack([s, g])

import agents.ModelTorch as ModelTorch

def main():
    global cfg
    print(cfg)
    counter = 0
    while True:
        encoder = Normalizer(len(transform(ENV.reset())) * cfg['history_count'] + cfg['her_state_size'])
        encoder_s = Normalizer(len(transform(ENV.reset())))
        encoder_g = Normalizer(3)

        counter += 1
        bot = Zer0Bot(
            cfg,
#            FetchNReachTask(cfg, (encoder_s, encoder_g)), # task "manager"
            FetchNReachTask(cfg, encoder), # task "manager"
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
        for _ in range(10): print("total reward : ", sum(bot.task_main.test_policy(bot, False)[2]))
        while True: bot.task_main.test_policy(bot, True)
        break

if '__main__' == __name__:
    main()
