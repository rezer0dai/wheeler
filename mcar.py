
# coding: utf-8

# In[1]:

import os, time


import numpy as np
import toml, gym

import torch
from torch.multiprocessing import Queue, Process


# In[2]:

CFG = toml.loads(open('cfg.toml').read())
GYM_CFG = toml.loads(open('gym.toml').read())

torch.set_default_tensor_type(CFG['tensor'])


# In[3]:

from utils.task import Task
from utils.taskinfo import *

from utils.rbf import *
from utils.normalizer import *

from utils.taskmgr import *
from utils.replay import *

from utils.fastmem import Memory

from utils.curiosity import *

from models.gru_model import CriticNN
from models.gru_model import ActorNN

from agent.zer0bot import agent_launch


# In[4]:

class MCarTask(Task):
    def __init__(self, cfg, env, objective_id, bot_id, action_low, action_high, rewarder):
        self.reward = 0
        self.rewards = []

        super().__init__(
                cfg,
                env,
                objective_id,
                bot_id,
                action_low, action_high)

        self.rewarder = rewarder

    def reset(self, seed = None, test = False):
        state = super().reset(seed, test)
        self.rewards.append(self.reward)
        self.reward = 0
        self.prev_state = state[0]
        return state

    def step_ex(self, action, test = False):
        state, reward, done, _ = self.env.step(self.bot_id, self.objective_id, action)

        if test: return action, state.reshape(1, -1), reward, done, True

        self.reward += (done and reward > 0)

        curiosity_test = '''
        if not test and reward < 0:
            ss, ns = np.vstack([s] * self.cfg['history_count']), np.vstack([n] * self.cfg['history_count'])
            ss = ss.reshape(1, -1)
            ns = ns.reshape(1, -1)
            reward = self.rewarder.weight(ss, ns, action)[0]
            self.rewarder.update(ss, ns, action)
        self.prev_state = state

        reward_update  = '''
        if not test and not done:# and sum(self.rewards) < 3
            true_state = np.abs(np.cos(np.pi/3.) + state[0])
            reward += -(1. - true_state)
#        '''

        return action, state, reward, done, True

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards))
        return sum(rewards) > 90.


# In[5]:

class MCarInfo(TaskInfo):
    def __init__(self, env, replaybuf, factory, Mgr, args):
        super().__init__(
                len(env.reset()), 1, -1, +1,
                CFG,
                replaybuf,
                factory, Mgr, args)

        self.rewarder = CuriosityPrio(
                self.state_size, self.action_size,
                self.action_range, self.wrap_action, "cpu", GYM_CFG)

    def new(self, cfg, bot_id, objective_id):
        return MCarTask(cfg,
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high,
                self.rewarder)

    @staticmethod
    def factory(ind): # bare metal task creation
        print("created %i-th task"%ind)
        CFG = toml.loads(open('cfg.toml').read())
        return gym.make(CFG['task'])


# In[6]:

def callback(task, agent, scores):
    try: callback.z += 1
    except: callback.z = 0

    print("WE GOT NEW SCORES!!", scores)

    done = all(task.test_policy(agent)[0] for _ in range(10))
    if not done:
        return False

    print("\n")
    print("="*80)
    print("training over", callback.z * GYM_CFG['n_simulations'] * GYM_CFG['mcts_rounds'])
    print("="*80)

    for i in range(100): print("total steps : training : %i :: %i >"%(
        callback.z * GYM_CFG['mcts_rounds'] * GYM_CFG['n_simulations'],
        len(task.test_policy(agent)[2])))

    return True

# In[7]:

def main():
    print(CFG)

    env = gym.make(CFG['task'])
    encoder = RbfState(env, [5., 2., 1., .5], [20] * 4)
    info = MCarInfo(env, Memory, MCarInfo.factory, LocalTaskManager, ())
#    info = MCarInfo(env, ReplayBuffer, MCarInfo.factory, RemoteTaskManager, (LocalTaskManager, 1 + GYM_CFG['n_simulations']))

    task = info.new(GYM_CFG, 0, -1)
    def callback_task(agent, stop_q):
        return callback(task, agent, stop_q)

    norm = GlobalNormalizer(encoder.out_size())
    encoder_norm = RBFEncoderWithNormalization(len(env.reset()), encoder, norm)

    stop_q = Queue()
    agent_launch(0, GYM_CFG, info, encoder_norm, ActorNN, CriticNN, stop_q, callback_task)

if '__main__' == __name__:
    main()


# In[ ]:



