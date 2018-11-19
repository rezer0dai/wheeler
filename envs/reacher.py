
# coding: utf-8

# # MountainCarContinuous-v0 environment of OpenAi GYM 
# - *Wheeler task definition ( task wrapper, State decoder settings, NeuralNetwork, ReplayBuffer, .. )*

# ### Import generics

# In[1]:


import os, time

import numpy as np
import toml, gym

import torch
from torch.multiprocessing import Queue, Process


# ### Load task configs ~ this should be adopted offline for particular task

# In[2]:


CFG = toml.loads(open('cfg.toml').read())
GYM_CFG = toml.loads(open('gym.toml').read())

torch.set_default_tensor_type(CFG['tensor'])
print(CFG['task'])
CFG['task'] = "unity/Reacher_Linux_20/Reacher.x86_64"


# ### Import wheeler environment and particular utils we want to use ~ general ones ( shared across tasks )

# In[3]:


from utils.task import Task
from utils.taskinfo import *

from utils.rbf import *
from utils.normalizer import *

from utils.taskmgr import *
from utils.replay import *

from utils.fastmem import Memory

from utils.curiosity import *

from agent.zer0bot import agent_launch

from utils.unity import unity_factory

# ### Define Task wrapper ~ when is goal met, how to step ( update rewards function, .. ), when / how to reset

# In[4]:


class GymTask(Task):
    def reset(self, seed = None, test = False):
        cfg = {"goal_size":5., "goal_speed":1.}
        state = super().reset(cfg, test)[0]

        if test: return state # we will get array of states

        return [state.reshape(-1)]
    
    def step_ex(self, action, test = False):
        state, done, reward = self.env.step(self.bot_id, self.objective_id, action)
        return action, state, reward, done, True

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards), sum(map(lambda r: r != 0, rewards)), len(rewards))
        return sum(abs(r) for r in rewards) > 30


# ### Generic proxy for creating our Task ( multiprocess environments purpose mainly ) 
# - but can also add wrapping function approx values ( action value to tanh, sigmoid, .. ) - this not works well with PPO now

# In[5]:


class GymInfo(TaskInfo):
    def __init__(self, replaybuf, factory, Mgr, args):
        super().__init__(
                33, 4, -1, +1,
                CFG,
                replaybuf,
                factory, Mgr, args)

    def new(self, cfg, bot_id, objective_id):
        task = GymTask(cfg,
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high)
        if -1 == objective_id:
            task.reset()
        return task

# ### Implement callback for testing policy ~ per X training rounds, we want to test it ~ enable visuals if you want

# In[6]:


def callback(task, agent, scores):
    try: callback.z += 1
    except: callback.z = 0
    
    # we can save scores to main queue, and avarage them, or we can ..
    # run testing w/ visuals :
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


# ### Prepare neural network which we will be using

# In[7]:


from models import ddpg_model, noisy_model, state_action_model

def CriticNN(state_size, action_size, wrap_value, cfg):
    return state_action_model.Critic(state_size, action_size, wrap_value, cfg, fcs1_units=256, fc2_units=128)
    return ddpg_model.Critic(state_size, action_size, wrap_value, cfg, fcs1_units=400, fc2_units=300)

def ActorNN(state_size, action_size, wrap_action, cfg):
    return noisy_model.Actor(state_size, action_size, wrap_action, cfg, hiddens=[128, 64])
    return ddpg_model.Actor(state_size, action_size, wrap_action, cfg, fc1_units=400, fc2_units=300)


# ### Select encoders

# In[8]:


from utils.encoders import *
from utils.rnn import *#GRUEncoder

def encoderstack():
    norm = GlobalNormalizer(GYM_CFG, 33)
    return norm
    experience = GRUEncoder(GYM_CFG, norm.total_size())#GRU#LSTM
    encoder_norm = StackedEncoder(GYM_CFG, 33, norm, experience)
    encoder_norm.share_memory()
    return encoder_norm


# ### Cook Task : replay buffer ( fast / prio-gae-rnn ) + task manager ( local / remote / unity )

# In[9]:


def taskfactory():
#    return GymInfo(Memory, GymInfo.factory, LocalTaskManager, ())
#    return GymInfo(ReplayBuffer, GymInfo.factory, LocalTaskManager, ())
    return GymInfo(Memory, unity_factory(CFG, CFG['total_simulations']), RemoteTaskManager, (LocalTaskManager, 1 + GYM_CFG['n_simulations']))
    return GymInfo(ReplayBuffer, unity_factory(CFG, CFG['total_simulations']), RemoteTaskManager, (LocalTaskManager, 1 + GYM_CFG['n_simulations']))


# ### Glue it all together ~ select buffer, encoders, agents, ... and RUN!!

# In[ ]:


def main():
    print(CFG)
    
    encoder = encoderstack()
    task_factory = taskfactory()
    task = task_factory.new(GYM_CFG, 0, -1)
    
    def callback_task(agent, stop_q):
        return callback(task, agent, stop_q)
    
    stop_q = Queue()
    agent_launch(0, GYM_CFG, task_factory, encoder, ActorNN, CriticNN, stop_q, callback_task)

if '__main__' == __name__:
    main()

