import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import numpy as np
import toml, torch, gym

from utils.rbf import *
from utils.task import Task
from utils.taskinfo import *
from utils.taskmgr import *
from utils.normalizer import *

from agents.zer0bot import Zer0Bot
import agents.ModelTorch as ModelTorch

from utils.curiosity import *
from utils.replay import *

def transform(s):
    return np.hstack([
        np.arccos(s[0]),
        np.arccos(s[2]),
        s,
        ])

def goal_select(total_after, n_step):
    if total_after <= n_step + 1: # only last n_state remainds
        return 0
    if random.randint(0, 2):
        return random.randint(1, n_step)
    return random.randint(1, total_after - 1 - n_step)

def fun_reward(s, n, gs, objective_id, cfg, her):
    state_size = (len(s) - cfg['her_state_size']) // cfg['history_count']
    s_ = s[-state_size:]
    return -1 * int(-np.cos(_s[1]) - np.cos(_s[2]+_s[1]) <= gs[0])

def sample_goal(goal, target, n_target, cfg):
    state_size = (len(s) - cfg['her_state_size']) // cfg['history_count']
    def noise_goal():
        g = goal[-state_size:]
        return [-np.cos(g[1]) - np.cos(g[2]+g[1]) - 1e-3]
    gs = noise_goal()
    return (np.hstack([gs, target[1:]]).reshape(-1),
            np.hstack([gs, n_target[1:]]).reshape(-1))

class AcroBotTask(Task):
    def __init__(self, 
            cfg, env, 
            objective_id, bot_id, 
            action_low, action_high, state_size, 
            encoder):

        super().__init__(
                cfg,
                env,
                objective_id,
                bot_id,
                action_low, action_high)

        self.state_size = state_size
        self.encoder = encoder

    def reset(self, seed = None, test = False):
        state = super().reset(seed)
        state = transform(state[0])
        return [state.reshape(-1)]

    def step_ex(self, action, test = False):
        action, a = self.softmax_policy(action, test)
        state, reward, done, _ = self.env.step(self.bot_id, self.objective_id, a)

        state = transform(state)

        if test: state = [state] 
        return action, state, reward, done, True

    def goal_met(self, states, rewards, n_steps):
        print("TEST ", sum(rewards))
        return sum(rewards) > -130.

    def her_state(self, _):
        return [1.]

    def normalize_state__(self, states):
        states = np.array(states).reshape(
                -1, 
                self.state_size * self.cfg['history_count'] + self.cfg['her_state_size'])
        return self.encoder.normalize(states)

    def update_normalizer__(self, states):
        states = np.vstack(states)
        self.encoder.update(states)

    def update_goal(self, _, states, n_states, updates):
        rews = []
        stat = []
        nstat = []
        for i, (n, s, u) in enumerate(zip(states, n_states, updates)):
            if u:
                g = states[i + goal_select(len(states) - i, self.cfg['n_step'])]
                s, n = sample_goal(g, s, n, self.cfg)
            else:
                g = s[:1]
            stat.append(s)
            nstat.append(n)
            rews.append(fun_reward(s, n, g, self.objective_id, self.cfg, True))
        return (rews, stat, nstat)

    def wrap_action(self, x):
        return x

class AcroBotInfo(TaskInfo):
    def __init__(self, cfg, state_size, encoder, replaybuf, factory, Mgr, args):
        super().__init__(
                state_size, 2, 0, 1,
                cfg,
                encoder, replaybuf,
                factory, Mgr, args)


    def new(self, cfg, bot_id, objective_id):
        return AcroBotTask(cfg, 
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high, self.state_size, self.encoder)

    @staticmethod
    def factory(ind): # bare metal task creation
        print("created %i-th task"%ind)
        CFG = toml.loads(open('cfg.toml').read())
        return gym.make(CFG['task'])

def main():
    CFG = toml.loads(open('cfg.toml').read())
    torch.set_default_tensor_type(CFG['tensor'])

    print(CFG)

    ENV = gym.make(CFG['task'])
    state_size = len(transform(ENV.reset()))
    DDPG_CFG = toml.loads(open('gym.toml').read())
    encoder = Normalizer(
            DDPG_CFG['history_count'] * state_size + DDPG_CFG['her_state_size'])

    INFO = AcroBotInfo(
            CFG, state_size, 
            encoder, 
            ReplayBuffer, 
            AcroBotInfo.factory, LocalTaskManager, ())

    task = INFO.new(DDPG_CFG, 0, -1)

    bot = Zer0Bot(
        0,
        DDPG_CFG,
        INFO,
        ModelTorch.ActorNetwork,
        ModelTorch.CriticNetwork)

    bot.turnon()

    z = 0
    task.training_status(False)
    while not task.learned():
        bot.train()
        print()
        task.training_status(
                all(task.test_policy(bot)[0] for _ in range(10)))
        z+=1

    print("\n")
    print("="*80)
    print("training over", z * CFG['n_simulations'] * CFG['mcts_rounds'])
    print("="*80)

    for i in range(10): print("total steps : training : %i :: %i >"%(
        z * CFG['mcts_rounds'] * CFG['n_simulations'],
        len(task.test_policy(bot)[2])))

if '__main__' == __name__:
    main()
