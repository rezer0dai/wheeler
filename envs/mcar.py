import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))

import numpy as np
import toml, torch, gym

from utils.rbf import *
from utils.task import Task
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
        self.prev_state = state
        return self.encoder.transform(state)

    def step_ex(self, action, test = False):
        self.n_steps += 1
        state, reward, done, _ = self.env.step(self.bot_id, self.objective_id, action)#self.env.step(action)#
#        action = clamped_action # if we clamp this is problem for PPO ...

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

class TaskInfo:
    def __init__(self, cfg, encoder, Mgr, args):
        self.state_size = 80#env.observation_space.shape[0]
        self.action_size = 1#env.action_space.shape[0]
        self.action_low = -1
        self.action_high = +1

        self.action_range = self.action_high - self.action_low

        self.env = Mgr(self._new, *args)
        self.cfg = cfg
        self.encoder = encoder

        self.rewarder = CuriosityPrio(
                self.state_size, self.action_size,
                self.action_range, self.wrap_action, "cpu", cfg)

    def wrap_value(self, x):
        return torch.clamp(x, min=self.cfg['min_reward_val'], max=self.cfg['max_reward_val'])

    def wrap_action(self, x):
        return torch.clamp(x, min=self.action_low, max=self.action_high)

    def new(self, bot_id, objective_id):
        return MCarTask(self.cfg, self.encoder, 
                self.env,
                objective_id, bot_id,
                self.action_low, self.action_high,
                self.rewarder)

    def make_replay_buffer(self, objective_id):
        buffer_size = self.cfg['replay_size']
        return ReplayBuffer(self.cfg, objective_id)

    @staticmethod
    def _new(ind): # bare metal task creation
        print("created %i-th task"%ind)
        CFG = toml.loads(open('cfg.toml').read())
        return gym.make(CFG['task'])

def main():
    CFG = toml.loads(open('cfg.toml').read())
    torch.set_default_tensor_type(CFG['tensor'])

    print(CFG)

    ENV = gym.make(CFG['task'])
    ENCODER = RbfState(ENV, [5., 2., 1., .5], [20] * 4)
    INFO = TaskInfo(CFG, ENCODER, LocalTaskManager, ())
#    INFO = TaskInfo(CFG, ENCODER, RemoteTaskManager, (LocalTaskManager, 1 + CFG['n_simulations']))
    task = INFO.new(0, -1)

    counter = 0
    while True:
        counter += 1
        bot = Zer0Bot(
            0,
            CFG,
            INFO,
            ModelTorch.ActorNetwork,
            ModelTorch.CriticNetwork)

        bot.start()

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

        break

if '__main__' == __name__:
    main()
