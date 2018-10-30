import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import numpy as np
import random, toml, torch, gym

from utils.normalizer import *
from utils.task import Task
from utils.taskmgr import *
from utils.crossexp import *

from agents.zer0bot import Zer0Bot
import agents.ModelTorch as ModelTorch

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

def fun_reward(s, n, gs, objective_id, cfg, her):
    hs = cfg['her_state_size']
    ind = -(len(n) - hs) // cfg['history_count']

    # dont divide 3D navigation to subtasks (x, y, z)
    return -1 * (2 * CLOSE_ENOUGH < np.abs(goal_distance(s[ind:ind+hs], gs[:hs])))

    xid = objective_id % 4
    if xid >= 3: # full task 3D naigation
        return -1 * (2 * CLOSE_ENOUGH < np.abs(goal_distance(s[ind:ind+hs], gs[:hs])))
# X- Y - Z subtask navigation
    a = np.abs(n[ind+xid] - gs[xid])
    b = np.abs(s[ind+xid] - gs[xid])
    if b < CLOSE_ENOUGH:
        return 0.
    return -1 + .9 * int(a < b)

def sample_goal(cfg, goal, target, n_target):
    def noisy_goal():
        hs = cfg['her_state_size']
        ind = -(len(goal) - hs) // cfg['history_count']
#        return goal[ind:ind+hs]
        for i in range(3):# be carefull extremly expensive
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
    gs = noisy_goal()
    return (np.hstack([gs, target[3:]]).reshape(-1),
            np.hstack([gs, n_target[3:]]).reshape(-1))

def goal_select(total_after, n_step):
    if total_after <= n_step + 1: # only last n_state remainds
        return 0
    if random.randint(0, 2):
        return random.randint(1, n_step)
    return random.randint(1, total_after - 1 - n_step)

class Reacher(Task):
    def __init__(self, cfg, encoder, env, objective_id, bot_id, action_low, action_high, state_size):
        self.encoder = encoder # well disable this until dont sync critic + actor to have same ..
        self.lock = threading.RLock()
        self.state_size = state_size

        self.n_step = cfg['n_step']

        super().__init__(
                cfg,
                env,
                objective_id,
                bot_id,
                action_low, action_high)

    def reset(self, seed = None, test = False):
        cfg = {"goal_size":CLOSE_ENOUGH * 4, "goal_speed":1.}
        state = super().reset(cfg, test)

        self.goal = state[-4-3:-1-3]
        state = transform(state)
        self.prev_state = np.zeros(len(state))
        return state.reshape(-1)

    def update_goal(self, _, states, n_states, updates):
        rews = []
        stat = []
        nstat = []
        for i, (n, s, u) in enumerate(zip(states, n_states, updates)):
            if u:
                g = states[i + goal_select(len(states) - i, self.n_step)]
                s, n = sample_goal(self.cfg, g, s, n)
            else:
                g = s[:self.cfg['her_state_size']]
            stat.append(s)
            nstat.append(n)
            rews.append(fun_reward(s, n, g, self.objective_id, self.cfg, True))
        return (rews, stat, nstat)


    def step_ex(self, action, test = False):
        self.n_steps += 1
        
        if test: action = [action] * self.cfg['task_required_simulations'] # well this is obiousely bad lol ...
        state, done, r = self.env.step(self.bot_id, self.objective_id, action)

#        print("given reward for objective", r, self.objective_id)

        self.goal = state[-4-3:-1-3]
        state = transform(state)
        good = True

#        reward = r if test else -(r == 0)

        reward = r if test else fun_reward(np.hstack(
            [self.goal, np.hstack([self.prev_state] * self.cfg['history_count'])]),
                np.hstack([self.goal, np.hstack([state] * self.cfg['history_count'])]),
                self.goal, self.objective_id, self.cfg, False)

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

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards), sum(map(lambda r: r != 0, rewards)), len(rewards))
        return sum(abs(r) for r in rewards) > 30

    def her_state(self):
        return self.goal

# move those to to replay buffer manger .. so at first create replay buffer manager lol ...
    def normalize_state(self, states): # here is ok to race ~ well not ok but i dont care now :)
        states = np.array(states).reshape(-1, 
                self.state_size * self.cfg['history_count'] + self.cfg['her_state_size'])

        s = self.encoder[0].normalize(states[:, self.cfg['her_state_size']:])
        g = self.encoder[1].normalize(states[:, :self.cfg['her_state_size']])
        return np.hstack([s, g])

# need to share params and need to be same as all simulations using!! -> move to TaskInfo!!
    def update_normalizer(self, states):
        states = np.vstack(states)
        with self.lock:
#            self.encoder.update(states)
            self.encoder[0].update(states[:, 3:])
            self.encoder[1].update(states[:, :3])
            self.encoder[1].update(states[:, 3:6])

class TaskInfo:
    def __init__(self, cfg, encoder, factory, Mgr, args):
        self.state_size = 30 - 4#env.observation_space.shape[0]
        self.action_size = 4#env.action_space.shape[0]
        self.action_low = -1
        self.action_high = +1

        self.action_range = self.action_high - self.action_low

        self.env = Mgr(factory, *args)
        self.cfg = cfg
        self.encoder = encoder

        self.CBE = cross_exp_buffer(self.cfg)

    def wrap_value(self, x):
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.clamp(x, min=self.action_low, max=self.action_high)

    def new(self, cfg, bot_id, objective_id):
        return Reacher(cfg, self.encoder, 
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high, self.state_size)

    def make_replay_buffer(self, cfg, objective_id):
        buffer_size = cfg['replay_size']
        return self.CBE(cfg, objective_id)

def main():
    CFG = toml.loads(open('cfg.toml').read())
    torch.set_default_tensor_type(CFG['tensor'])

    print(CFG)

    encoder_s = Normalizer((30-4) * CFG['history_count'])
    encoder_s.share_memory()
    encoder_g = Normalizer(CFG['her_state_size'])
    encoder_g.share_memory()
    encoder = (encoder_s, encoder_g)#None#

    from utils.unity import unity_factory

    INFO = TaskInfo(
            CFG, 
            encoder, 
            unity_factory(CFG['task_required_simulations']), 
            RemoteTaskManager, (LocalTaskManager, 1 + CFG['task_required_simulations']))

    task = INFO.new(CFG, 0, -1)
    task.reset()

    bot_ddpg = Zer0Bot(
        0,
        CFG,
        INFO,
        ModelTorch.ActorNetwork,
        ModelTorch.CriticNetwork)

#    enable_two_bots = '''
    # basic idea of this, is that PPO will greedy explore current policy, while DDPG will benefit
    # from its experience ~ PPO no necessary to learn properly, target is DDPG
    # we want also to load PPO model from DPPG one periodically, as so using PPO as explorer of close unknown
    PPO_CFG = toml.loads(open('ppo_cfg.toml').read())
    assert PPO_CFG['max_n_episode'] == CFG['max_n_episode'], "max episode count must be equal for all agents!"
    bot_ppo = Zer0Bot(
        1,
        PPO_CFG,
        INFO,
        ModelTorch.ActorNetwork,
        ModelTorch.CriticNetwork)
    bot_ppo.turnon() # launch critics!
    bot_ppo.start() # ppo will run at background
#    '''

    bot_ddpg.turnon()

    task.training_status(
            all(task.test_policy(bot_ddpg)[0] for _ in range(10)))

    z = 0
    task.training_status(False)
    while not task.learned():
        bot_ddpg.train()

        bot_ppo.actor.model.beta_sync() # update our explorer

        print()
        task.training_status(
                all(task.test_policy(bot_ddpg)[0] for _ in range(10)))
        z+=1

    bot_ppo.stop.put(True)

    print("\n")
    print("="*80)
    print("training over", counter, z * CFG['n_simulations'] * CFG['mcts_rounds'])
    print("="*80)

    while not bot_ppo.stop.empty():
        pass

if '__main__' == __name__:
    main()
