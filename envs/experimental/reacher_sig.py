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

class Reacher(Task):
    def __init__(self, cfg, env, objective_id, bot_id, action_low, action_high, state_size):
        super().__init__(
                cfg,
                env,
                objective_id,
                bot_id,
                action_low, action_high)

    def reset(self, seed = None, test = False):
        cfg = {"goal_size":5., "goal_speed":1.}
        state = super().reset(cfg, test)[0]

        if test: return state # we will get array of states

        return [state.reshape(-1)]

    def step_ex(self, action, test = False):
        state, done, reward = self.env.step(self.bot_id, self.objective_id, action)

        if test:
            return action, state, reward, done, True

        self.n_steps += 1
        return action, state, reward, done, True

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards), sum(map(lambda r: r != 0, rewards)), len(rewards))
        return sum(abs(r) for r in rewards) > 30

class ReacherInfo(TaskInfo):
    def __init__(self, cfg, replaybuf, factory, Mgr, args):
        super().__init__(
                33, 4, -1, +1,
                cfg,
                None, replaybuf,
                factory, Mgr, args)

    def new(self, cfg, bot_id, objective_id):
        return Reacher(cfg,
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high, self.state_size)

def main():
    CFG = toml.loads(open('cfg.toml').read())
    torch.set_default_tensor_type(CFG['tensor'])

#    BOT_CFG = toml.loads(open('ddpg_cfg_sig.toml').read())
    BOT_CFG = toml.loads(open('ppo_cfg_sig.toml').read())

    from utils.unity import unity_factory

    INFO = ReacherInfo(
            CFG, 
            ReplayBuffer,#cross_exp_buffer(CFG),#
            unity_factory(CFG, CFG['total_simulations']), 
            RemoteTaskManager, (LocalTaskManager, CFG['total_simulations']))

    task = INFO.new(BOT_CFG, 0, -1)
    task.reset(None, True)

    bot = Zer0Bot(
        0,
        BOT_CFG,
        INFO,
        ModelTorch.ActorNetwork,
        ModelTorch.CriticNetwork)

    bot.turnon()

    task.training_status(
            all(task.test_policy(bot)[0] for _ in range(10)))

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
    print("training over", counter, z * BOT_CFG['n_simulations'] * BOT_CFG['mcts_rounds'])
    print("="*80)

if '__main__' == __name__:
    main()
