import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import numpy as np
import toml, torch, gym

from utils.task import Task
from utils.taskinfo import *
from utils.taskmgr import *

from agents.zer0bot import Zer0Bot
import agents.ModelTorch as ModelTorch

from utils.curiosity import *
from utils.replay import *

class BipedalTask(Task):
    def __init__(self, cfg, env, objective_id, bot_id, action_low, action_high):
        super().__init__(
                cfg,
                env,
                objective_id,
                bot_id,
                action_low, action_high)

    def step_ex(self, action, test = False):
        state, reward, done, _ = self.env.step(self.bot_id, self.objective_id, action)
        if test: return action, [ state.reshape(-1) ], reward, done, True
        return action, state, reward, done, True

    def goal_met(self, states, rewards, n_steps):
        print("TEST ", sum(rewards))
        return sum(rewards) > 300

class BipedalInfo(TaskInfo):
    def __init__(self, cfg, replaybuf, factory, Mgr, args):
        env = self.factory(0)
        super().__init__(
                len(env.reset()), 4, 
                float(env.action_space.low[0]), float(env.action_space.high[0]),
                cfg,
                None, replaybuf,
                factory, Mgr, args)


    def new(self, cfg, bot_id, objective_id):
        return BipedalTask(cfg, 
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high)

    @staticmethod
    def factory(ind): # bare metal task creation
        print("created %i-th task"%ind)
        CFG = toml.loads(open('cfg.toml').read())
        return gym.make(CFG['task'])

def main():
    CFG = toml.loads(open('cfg.toml').read())
    torch.set_default_tensor_type(CFG['tensor'])

    print(CFG)

    INFO = BipedalInfo(CFG, ReplayBuffer, BipedalInfo.factory, LocalTaskManager, ())

    DDPG_CFG = toml.loads(open('gym.toml').read())

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

