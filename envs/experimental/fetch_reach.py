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

 # not testedm, just ported Unity-Reacher here, should works with minor bug fixes i guess, TODO : test ...

CLOSE_ENOUGH = .05

def transform(obs):
    return np.hstack([
        obs["achieved_goal"],
        obs["observation"][3:],
        ])

import gym
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

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

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

            if np.abs(
                    gym.envs.robotics.fetch_env.goal_distance(
                        g[:3], g[3:2*3])) < CLOSE_ENOUGH:

                return g[:3]

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

class FetchReachGym(Task):
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

    def reset(self, seed = None, test = False):
        state = super().reset(seed)[0]
        self.goal = state["desired_goal"]
        state = transform(state)
        self.prev_state = np.zeros(len(state))
        return [state.reshape(-1)]

    def step_ex(self, action, test = False):
        state, done, reward = self.env.step(self.bot_id, self.objective_id, action)

        self.goal = state["desired_goal"]
        state = transform(state)
        good = True

        reward = fun_reward(np.hstack(
            [self.goal, np.hstack([self.prev_state] * self.cfg['history_count'])]),
                np.hstack([self.goal, np.hstack([state] * self.cfg['history_count'])]),
                self.goal, self.objective_id, self.cfg, False)

        self.prev_state = state
        if test: state = [state] 
        return action, state, reward, done, good

    def her_state(self, _):
        return self.goal

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
    def normalize_state(self, states): # here is ok to race ~ well not ok but i dont care now :)
        states = np.array(states).reshape(-1, 
                self.state_size * self.cfg['history_count'] + self.her_size)

        s = self.encoder[0].normalize(states[:, self.her_size:])
        g = self.encoder[1].normalize(states[:, :self.her_size])
        return np.hstack([s, g])

# need to share params and need to be same as all simulations using!! -> move to TaskInfo!!
    def update_normalizer(self, states):
        states = np.vstack(states)
        with self.lock:
#            self.encoder.update(states)
            self.encoder[0].update(states[:, self.her_size:])
            self.encoder[1].update(states[:, :self.her_size])
            self.encoder[1].update(states[:, self.her_size:self.her_size*2])

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards), sum(map(lambda r: r != 0, rewards)), len(rewards))
        return -5 < sum(rewards[len(rewards)//2:])

class FetchReachGymInfo(TaskInfo):
    def __init__(self, cfg, encoder, replaybuf, factory, Mgr, args):
        state_size = len(ENV.reset())
        super().__init__(
                state_size, 4, -1, +1,
                cfg,
                encoder, replaybuf,
                factory, Mgr, args)

    def new(self, cfg, bot_id, objective_id):
        return FetchReachGym(cfg, self.encoder, 
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high, self.state_size)
    @staticmethod
    def factory(ind): # bare metal task creation
        print("created %i-th task"%ind)
        CFG = toml.loads(open('cfg.toml').read())
        return gym.make(CFG['task'])

def load_encoder(cfg, state_size):
    encoder_s = Normalizer(state_size * cfg['history_count'])
    encoder_s.share_memory()
    encoder_g = Normalizer(cfg['her_state_size'])
    encoder_g.share_memory()

    if os.path.exists("encoder_s.torch"):
        encoder_s.load_state_dict(torch.load("encoder_s.torch"))
        encoder_g.load_state_dict(torch.load("encoder_g.torch"))

    return (encoder_s, encoder_g)

def save_encoder(encoder):
    torch.save(encoder[0].state_dict(), "encoder_s.torch")
    torch.save(encoder[1].state_dict(), "encoder_g.torch")

def main():
    CFG = toml.loads(open('cfg.toml').read())
    torch.set_default_tensor_type(CFG['tensor'])

    print(CFG)

    DDPG_CFG = toml.loads(open('ddpg_cfg.toml').read())

    state_size = len(ENV.reset())
    encoder = load_encoder(DDPG_CFG, state_size)

    INFO = FetchReachGymInfo(
            CFG, 
            encoder, 
            cross_exp_buffer(CFG),
            FetchReachGymInfo.factory,
            RemoteTaskManager, (LocalTaskManager, CFG['total_simulations']))

    task = INFO.new(DDPG_CFG, 0, -1)
    task.reset(None, True)

    bot_ddpg = Zer0Bot(
        0,
        DDPG_CFG,
        INFO,
        ModelTorch.ActorNetwork,
        ModelTorch.CriticNetwork)

    enable_two_bots = '''
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

        save_encoder(encoder)

        print()
        task.training_status(
                all(task.test_policy(bot_ddpg)[0] for _ in range(10)))
        z+=1

    for bot in explorers:
        bot.stop.put(True)

    print("\n")
    print("="*80)
    print("training over", counter, z * DDPG_CFG['n_simulations'] * DDPG_CFG['mcts_rounds'])
    print("="*80)

    for bot in explorers:
        while not bot.stop.empty():
            pass

if '__main__' == __name__:
    main()
