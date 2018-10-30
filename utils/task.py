import torch
import torch.nn.functional as F

import abc, random
import numpy as np

from utils.replay import ReplayBuffer
from collections import deque

class Task(object, metaclass=abc.ABCMeta):
    ep_count = 0
    training_finished = False

    def __init__(self, cfg, env, objective_id, bot_id, action_low, action_high):
        self.cfg = cfg
        self.env = env
        self.bot_id = bot_id
        self.objective_id = objective_id

        self.action_low = action_low
        self.action_high = action_high
# globals
    def iter_count(self):
        return Task.ep_count
    def episode_counter(self):
        return Task.ep_count
    def learned(self):
        return Task.training_finished
    def training_status(self, status):
        Task.training_finished = status
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
# workers
    def make_replay_buffer(self, cfg):
        buffer_size = self.cfg['replay_size']
        return ReplayBuffer(cfg, self.objective_id)

    def step(self, action):
        r = 0
        for i, act in enumerate(action):
            for _ in range(self.cfg['action_repeat']):
                Task.ep_count += int(0 == self.objective_id)
                act = np.clip(act, self.action_low, self.action_high)
                a, state, reward, done, good = self.step_ex(act)
                r += reward
                if done:
                    break
            if any(a != act): # user want to change this action ( wrt gradient for training )
                print("#"*100, "ACTION CHANGED", action[i], a, act) # debug for test cartpole + acro
                action[i] = a
            #self.prev_state = self.env.pose
            if done:
                break
        return np.array(action), state, r, done, good

    def test_policy(self, bot, render = True):
        history = deque(maxlen=self.cfg['history_count'])
        state = self.reset(None, True)
        for s in [np.zeros(len(state))] * self.cfg['history_count']:
            history.append(np.vstack(s))

        states = []
        rewards = []
        h = np.zeros(shape=(1, 1, self.cfg['history_features']))
        while True:
            history.append(np.vstack(state))
            state = np.vstack(history).squeeze(1)
            state = self.transform_state(state)
            norm_state = self.normalize_state(state.copy())

            a, h = bot.act(norm_state, h)

#we are concerned only at first action, in test we dont repeat!
            a = a[0]
            h = h[0]

            if render:
                self.env.render()

            a = np.clip(a, self.action_low, self.action_high)
            _, state, reward, done, _ = self.step_ex(a, True)

            states.append(state)
            rewards.append(reward)
            if done:
                break

        return self.goal_met(states, rewards, len(rewards)), states, rewards

    def softmax_policy(self, action, test):
        a = np.argmax(action)

#        print(action)

        if 0 == random.randint(0, 1):#len(action)):#1):#
            return action, a

        try:
            aprob = torch.softmax(torch.from_numpy(action.reshape(-1)), 0)
        except:
            aprob = torch.nn.functional.softmax(torch.from_numpy(action.reshape(-1)), 0)

        a = np.random.choice(len(action), 1, p=aprob).item()
        action = np.zeros(action.shape)
        action[a] = 1.
        return action, a

    def her_state(self):
        return []

    def transform_state(self, state):
        return np.hstack([self.her_state(), state])

    def normalize_state(self, state):
# different than encoding
# why ? because this is running normalization
# it changing, and change also reasonability with state
# aka recalculating reward is after aplying this almost impossible
# ~ ok possible if this norm is reversible ofc ...
        return state

    def update_normalizer(self, states):
        return

    def reset(self, seed = None, test = False):
        if None == seed:
            seed = random.randint(0, self.cfg['mcts_random_cap'])
        self.n_steps = 0
        self._seed = seed
        return self.env_reset(seed, test)

    def seed(self):
        return self._seed

    def env_reset(self, seed, _):
        self.env.seed(self._seed)
        state = self.env.reset()
        return state

    def update_goal(self, rewards, states, updates):
        return rewards

    @abc.abstractmethod
    def goal_met(self, state, n_steps):
        pass

    @abc.abstractmethod
    def step_ex(self, action):
        pass
