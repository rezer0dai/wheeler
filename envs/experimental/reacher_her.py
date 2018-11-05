import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import numpy as np
import random, toml, torch, gym

from utils.normalizer import *
from utils.task import Task
from utils.taskinfo import *
from utils.taskmgr import *
from utils.crossexp import *

from agents.zer0bot import Zer0Bot
import agents.ModelTorch as ModelTorch

from utils.unity import unity_factory

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
    hs = cfg['her_state_size']
    def noisy_goal():
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
    return (np.hstack([gs, target[hs:]]).reshape(-1),
            np.hstack([gs, n_target[hs:]]).reshape(-1))

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
        self.her_size = cfg['her_state_size']

        super().__init__(
                cfg,
                env,
                objective_id,
                bot_id,
                action_low, action_high)

    def _extract_goal(self, state):
        return state[-4-3:-1-3]

    def _her_context_update(self, state):
        goal = []
        states = []
        for i in range(len(state)):
            goal.append(self._extract_goal(state[i]))
            states.append(transform(state[i]).reshape(-1))
        return goal, states

    def reset(self, seed = None, test = False):
        cfg = {"goal_size":CLOSE_ENOUGH * 4, "goal_speed":1.}
        state = super().reset(cfg, test)[0]

        if test:
            self.goal, states = self._her_context_update(state)
            return states

        self.goal = [ self._extract_goal(state) ]
        state = transform(state)
        self.prev_state = np.zeros(len(state))
        return [state.reshape(-1)]

    def her_state(self, i = 0):
        return self.goal[i]

    def update_goal(self, _, states, n_states, updates):
        rews = []
        stat = []
        nstat = []
        for i, (n, s, u) in enumerate(zip(states, n_states, updates)):
            if u:
                g = states[i + goal_select(len(states) - i, self.n_step)]
                s, n = sample_goal(self.cfg, g, s, n)
            else:
                g = s[:self.her_size]
            stat.append(s)
            nstat.append(n)
            rews.append(fun_reward(s, n, g, self.objective_id, self.cfg, True))
        return (rews, stat, nstat)

# move those to to replay buffer manger .. so at first create replay buffer manager lol ...
    def normalize_state__(self, states): # here is ok to race ~ well not ok but i dont care now :)
        states = np.array(states).reshape(-1, 
                self.state_size * self.cfg['history_count'] + self.her_size)

        s = self.encoder[0].normalize(states[:, self.her_size:])
        g = self.encoder[1].normalize(states[:, :self.her_size])
        return np.hstack([s, g])

# need to share params and need to be same as all simulations using!! -> move to TaskInfo!!
    def update_normalizer__(self, states):
        states = np.vstack(states)
        with self.lock:
#            self.encoder.update(states)
            self.encoder[0].update(states[:, self.her_size:])
            self.encoder[1].update(states[:, :self.her_size])
            self.encoder[1].update(states[:, self.her_size:self.her_size*2])

    def step_ex(self, action, test = False):
        state, done, reward = self.env.step(self.bot_id, self.objective_id, action)

        if test:
            self.goal, states = self._her_context_update(state)
            return action, states, reward, done, True

        self.n_steps += 1

        self.goal = [ self._extract_goal(state) ]
        state = transform(state)
        good = True

        reward = fun_reward(np.hstack(
            [self.goal[0], np.hstack([self.prev_state] * self.cfg['history_count'])]),
                np.hstack([self.goal[0], np.hstack([state] * self.cfg['history_count'])]),
                self.goal[0], self.objective_id, self.cfg, False)

        self.prev_state = state
        return action, state, reward, done, good

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards), sum(map(lambda r: r != 0, rewards)), len(rewards))
#        return -5 < sum(rewards[len(rewards)//2:])
        return sum(abs(r) for r in rewards) > 5#30

class ReacherInfo(TaskInfo):
    def __init__(self, cfg, encoder, replaybuf, factory, Mgr, args):
        super().__init__(
                30 - 4, 4, -1, +1,
                cfg,
                encoder, replaybuf,
                factory, Mgr, args)

    def new(self, cfg, bot_id, objective_id):
        return Reacher(cfg, self.encoder, 
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high, self.state_size)

def load_encoder(cfg):
    return
    encoder_s = Normalizer((30 - 4) * cfg['history_count'])
    encoder_s.share_memory()
    encoder_g = Normalizer(cfg['her_state_size'])
    encoder_g.share_memory()

    if os.path.exists("encoder_s.torch"):
        encoder_s.load_state_dict(torch.load("encoder_s.torch"))
        encoder_g.load_state_dict(torch.load("encoder_g.torch"))

    return (encoder_s, encoder_g)

def save_encoder(encoder):
    return
    torch.save(encoder[0].state_dict(), "encoder_s.torch")
    torch.save(encoder[1].state_dict(), "encoder_g.torch")

def main():
    CFG = toml.loads(open('cfg.toml').read())
    torch.set_default_tensor_type(CFG['tensor'])

    print(CFG)

    DDPG_CFG = toml.loads(open('ddpg_cfg.toml').read())
    encoder = load_encoder(DDPG_CFG)


    INFO = ReacherInfo(
            CFG, 
            encoder, 
            cross_exp_buffer(CFG),
            unity_factory(CFG, CFG['total_simulations']), 
            RemoteTaskManager, (LocalTaskManager, CFG['total_simulations']))

    task = INFO.new(DDPG_CFG, 0, -1)
    task.reset(None, True)

    bot_ddpg = Zer0Bot(
        0,
        DDPG_CFG,
        INFO,
        ModelTorch.ActorNetwork,
        ModelTorch.CriticNetwork)

#    enable_two_bots = '''
    PPO_CFG = toml.loads(open('ppo_cfg.toml').read())

    assert (
        PPO_CFG['max_n_episode'] == DDPG_CFG['max_n_episode'] and
        PPO_CFG['her_state_features'] == DDPG_CFG['her_state_features'] and
        PPO_CFG['history_features'] == DDPG_CFG['history_features'] and
        PPO_CFG['her_state_size'] == DDPG_CFG['her_state_size']
        ), "check assert those need to have be the same in order of PPO to boost DDPG"

    explorers = [ Zer0Bot(
        1 + i,
        PPO_CFG,
        INFO,
        ModelTorch.ActorNetwork,
        ModelTorch.CriticNetwork) for i in range(2) ]
    for bot in explorers:
        bot.turnon() # launch critics!
        bot.start() # ppo will run at background

    single_bot_setting = '''
    explorers = []
#    '''

    bot_ddpg.turnon()

    task.training_status(
            all(task.test_policy(bot_ddpg)[0] for _ in range(10)))

    z = 0
    while not task.learned():
        bot_ddpg.train()

        for bot in explorers: # we dont want to read last layer ~ that is for prob distrubution
            bot.actor.model.beta_sync(['ex']) # update our explorer
            # we want to proble DDPG by most of it network loaded but by PPO learned last layer

        save_encoder(encoder)

        print()
        task.training_status(
                all(task.test_policy(bot_ddpg)[0] for _ in range(10)))
        z+=1

    for bot in explorers:
        bot.stop.put(True)

    print("\n")
    print("="*80)
    print("training over", z * DDPG_CFG['n_simulations'] * DDPG_CFG['mcts_rounds'])
    print("="*80)

    for bot in explorers:
        while not bot.stop.empty():
            pass

if '__main__' == __name__:
    main()
