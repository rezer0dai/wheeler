import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import numpy as np
import toml, torch, gym

from utils.rbf import *
from utils.task import Task
from utils.taskinfo import *
from utils.taskmgr import *

from agents.zer0bot import Zer0Bot
import agents.ModelTorch as ModelTorch

from utils.curiosity import *
from utils.replay import *

class CartPoleTask(Task):
    def __init__(self, cfg, env, objective_id, bot_id, action_low, action_high):
        super().__init__(
                cfg,
                env,
                objective_id,
                bot_id,
                action_low, action_high)
        self.n_steps = 0
        self.max_n_episode = cfg['max_n_episode']

    def step_ex(self, action, test = False):
        self.n_steps += 1
        action, a = self.softmax_policy(action, test)
        state, reward, done, _ = self.env.step(self.bot_id, self.objective_id, a)
        if done:
            reward = -40 * int(self.n_steps < self.max_n_episode)#-40
            self.n_steps = 0
        if test: return action, [ state.reshape(-1) ], reward, done, True
        return action, state, reward, done, True

    def goal_met(self, states, rewards, n_steps):
        return sum(rewards) > 199 - 40

class CartPoleInfo(TaskInfo):
    def __init__(self, cfg, replaybuf, factory, Mgr, args):
        env = self.factory(0)
        super().__init__(
                len(env.reset()), 2, -2, +2,
                cfg,
                None, replaybuf,
                factory, Mgr, args)

    def new(self, cfg, bot_id, objective_id):
        return CartPoleTask(cfg, 
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high)

    def wrap_action(self, x):
#        return torch.tanh(x)
#        return x
#        return torch.sigmoid(x)
        return torch.clamp(x, min=-1, max=+2)

    @staticmethod
    def factory(ind): # bare metal task creation
        print("created %i-th task"%ind)
        CFG = toml.loads(open('cfg.toml').read())
        return gym.make(CFG['task'])

def main():
    CFG = toml.loads(open('cfg.toml').read())
    torch.set_default_tensor_type(CFG['tensor'])

    print(CFG)

    INFO = CartPoleInfo(CFG, ReplayBuffer, CartPoleInfo.factory, LocalTaskManager, ())

    DDPG_CFG = toml.loads(open('cart.toml').read())
#    DDPG_CFG = toml.loads(open('gym.toml').read())

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
    print("training over", z * DDPG_CFG['n_simulations'] * DDPG_CFG['mcts_rounds'])
    print("="*80)

    for i in range(10): print("total steps : training : %i :: %i >"%(
        z * DDPG_CFG['mcts_rounds'] * DDPG_CFG['n_simulations'],
        len(task.test_policy(bot)[2])))

if '__main__' == __name__:
    main()
