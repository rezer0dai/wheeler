import torch
import torch.nn.functional as F

import abc, random
import numpy as np

from utils.replay import ReplayBuffer
from collections import deque

class Task(object, metaclass=abc.ABCMeta):
    def __init__(self, cfg, env, objective_id, bot_id, action_low, action_high):
        self.cfg = cfg
        self.env = env
        self.bot_id = bot_id
        self.objective_id = objective_id

        self.action_low = action_low
        self.action_high = action_high

        self.env.register(self.bot_id, self.objective_id)
        self.ep_count = 0
# local info
    def name(self):
        return self.cfg['task']+"-def[%i]"%self.objective_id
    def device(self):
        if 0 == torch.cuda.device_count():
            return "cpu"
        if "cuda" not in self.cfg['device']:
            return self.cfg['device']
        if 1 == torch.cuda.device_count():
            return "cuda"
        return "cuda:%i"%(1 + (self.objective_id % (torch.cuda.device_count() - 1)))
    def iter_count(self):
        return self.ep_count

    def step(self, action):
        r = 0
        for i, act in enumerate(action):
            for _ in range(self.cfg['action_repeat']):
                self.ep_count += int(1 == self.objective_id)
                act = np.clip(act, self.action_low, self.action_high)
                a, state, reward, done, good = self.step_ex(act)
                r += reward
                if done:
                    break
            if any(a != act): # user want to change this action ( wrt gradient for training )
                action[i] = a
            if done:
                break
        return np.array(action), state, r, done, good

    def test_policy(self, bot):
        state = self.reset(None, True)

        history = [deque(maxlen=self.cfg['history_count']) for _ in range(len(state))]
        hiddens = []
        for h in history:
            for s in [np.zeros(len(state[0]))] * self.cfg['history_count']:
                h.append(np.vstack(s))
            hiddens.append(np.zeros(shape=(1, 1, self.cfg['history_features'])))

        rewards = []
        states = []
        done = 0
        while not np.sum(done):
            action = []
            for i, s in enumerate(state):
                history[i].append(np.vstack(s))
                state = np.vstack(history[i]).squeeze(1)
                goal = self.goal(i)

                a, h = bot.act(goal, state, hiddens[i])
                hiddens[i] = h[0]

                a = np.clip(a[0], self.action_low, self.action_high)
                action.append(a)

            _, state, reward, done, _ = self.step_ex(np.asarray(action).reshape(-1), True)

            states.append(state)
            rewards.append(np.mean(reward))

        return self.goal_met(states, rewards, len(rewards)), states, rewards

    def softmax_policy(self, action, test):
        a = np.argmax(action)

        if 0 == random.randint(0, 1):#len(action)):#
            return action, a

        action = np.asarray(action).reshape(-1)
        aprob = F.softmax(torch.from_numpy(action), 0)
        a = np.random.choice(len(action), 1, p=aprob).item()
        action = np.zeros(action.shape)
        action[a] = 1.
        return action, a

    def seed(self):
        return self._seed

    def update_goal(self, rewards, goals, states, n_goals, n_states, updates):
        return zip(rewards, goals, states, n_goals, n_states)

    def goal(self, ind = 0):
        return np.zeros(0)

    def reset(self, seed = None, test = False):
        if None == seed:
            seed = random.randint(0, self.cfg['mcts_random_cap'])
        self.n_steps = 0
        self._seed = seed
        return [self.env_reset(seed)]

    def env_reset(self, seed):
        return self.env.reset(self.bot_id, self.objective_id, seed)

    @abc.abstractmethod
    def goal_met(self, state, n_steps):
        pass

    @abc.abstractmethod
    def step_ex(self, action, test  = False):
        pass
