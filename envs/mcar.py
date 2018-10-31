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

class MCarTask(Task):
    def __init__(self, cfg, encoder, env, objective_id, bot_id, action_low, action_high, rewarder):
        self.encoder = encoder

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
        return [ self.encoder.transform(state[0]) ]

    def step_ex(self, action, test = False):
        self.n_steps += 1
        state, reward, done, _ = self.env.step(self.bot_id, self.objective_id, action)

        if test: return action, [ self.encoder.transform(state.reshape(-1)) ], reward, done, True

        self.reward += (done and reward > 0)

        curiosity_test = '''
        if not test and reward < 0:
            s = self.encoder.transform(self.prev_state)
            n = self.encoder.transform(state)
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

        return action, self.encoder.transform(state), reward, done, True

    def goal_met(self, states, rewards, n_steps):
        print("TEST : ", sum(rewards))
        return sum(rewards) > 90.

class MCarInfo(TaskInfo):
    def __init__(self, cfg, replaybuf,  encoder, factory, Mgr, args):
        super().__init__(
                80, 1, -1, +1,
                cfg,
                encoder, replaybuf,
                factory, Mgr, args)

        return
        self.rewarder = CuriosityPrio(
                self.state_size, self.action_size,
                self.action_range, self.wrap_action, "cpu", cfg)

    def new(self, cfg, bot_id, objective_id):
        return MCarTask(cfg, self.encoder, 
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high,
                None)#self.rewarder)

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
    ENCODER = RbfState(ENV, [5., 2., 1., .5], [20] * 4)
    INFO = MCarInfo(CFG, ReplayBuffer, ENCODER, MCarInfo.factory, LocalTaskManager, ())
#    INFO = TaskInfo(CFG, ENCODER, RemoteTaskManager, (LocalTaskManager, 1 + CFG['n_simulations']))

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
    print("training over", counter, z * CFG['n_simulations'] * CFG['mcts_rounds'])
    print("="*80)

    for i in range(10): print("total steps : %i < training : %i :: %i >"%(
        counter, 
        z * CFG['mcts_rounds'] * CFG['n_simulations'],
        len(task.test_policy(bot)[2])))

if '__main__' == __name__:
    main()
