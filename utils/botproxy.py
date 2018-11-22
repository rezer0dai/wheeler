import torch
from utils.bot import Bot

class BotProxy:
    def __init__(self, lock, cfg, bot, device):
        self.bot = bot
        self.cfg = cfg
        self.lock = lock
        self.device = device

        self.state_size = self.bot.state_size * cfg['history_count']

    def exploit(self, goals, states, history):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        goals = torch.DoubleTensor(torch.from_numpy(goals)).to(self.device) if all(s for s in goals.shape) else torch.zeros(0)

        with self.lock: # dist in DQN should be similiar proxy like utils.policy.DDPGDist
            dist, features = self.bot.exploit(goals, states, history)
            return (dist.sample().detach().cpu().numpy(),
                    features.detach().cpu().numpy())

    def explore(self, objective_id, goals, states, history):
        ind = objective_id if self.cfg['detached_actor'] else 0
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        goals = torch.DoubleTensor(torch.from_numpy(goals)).to(self.device) if all(s for s in goals.shape) else torch.zeros(0)

        with self.lock:
            return self.bot.explore(ind, goals, states, history)

    def qa_future(self, ind, goals, states, history, actions):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        actions = torch.DoubleTensor(torch.from_numpy(actions)).to(self.device)
        goals = torch.DoubleTensor(torch.from_numpy(goals)).to(self.device) if all(s for s in goals.shape) else torch.zeros(0)

        actions = actions.view(states.size(0), -1)
        assert states.size(0) == actions.size(0)

        with self.lock:
            return self.bot.qa_future(ind, goals, states, history, actions).detach().cpu().numpy()

    def qa_present(self, ind, goals, states, history, actions):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = history.view(1, states.size(0), -1).to(self.device)
        actions = actions.view(states.size(0), -1).to(self.device)
        goals = torch.DoubleTensor(torch.from_numpy(goals)).to(self.device) if all(s for s in goals.shape) else torch.zeros(0)

        with self.lock:
            return self.bot.qa_present(ind, goals, states, history, actions)

    def q_future(self, ind, goals, states, history):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        goals = torch.DoubleTensor(torch.from_numpy(goals)).to(self.device) if all(s for s in goals.shape) else torch.zeros(0)

        with self.lock:
            return self.bot.q_future(ind, goals, states, history).detach()

    def q_explore(self, objective_id, goals, states, history):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        goals = torch.DoubleTensor(torch.from_numpy(goals)).to(self.device) if all(s for s in goals.shape) else torch.zeros(0)

        a_ind = objective_id if self.cfg['detached_actor'] else 0
        with self.lock:
            return self.bot.q_explore(a_ind, objective_id, goals, states, history)

    def learn_actor(self, states, advantages, actions, tau):
        states = torch.DoubleTensor(states).to(self.device)
        actions = torch.DoubleTensor(actions).to(self.device)

        with self.lock:
            return self.bot.learn_actor(states, advantages, actions, tau)

    def learn_critic(self, ind, goals, states, history, actions, qa_target, tau):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        actions = torch.DoubleTensor(torch.from_numpy(actions)).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        goals = torch.DoubleTensor(torch.from_numpy(goals)).to(self.device) if all(s for s in goals.shape) else torch.zeros(0)

        with self.lock:
            return self.bot.learn_critic(ind, goals, states, history, actions, qa_target, tau)

    def alpha_sync(self, blacklist):
        self.bot.alpha_sync(blacklist)

    def reevaluate(self, objective_id, goals, states, actions):
        ind = objective_id if self.cfg['detached_actor'] else 0
        states = torch.from_numpy(states)
        goals = torch.from_numpy(goals) if all(s for s in goals.shape) else torch.zeros(0)
        with self.lock:
            return self.bot.reevaluate(ind, goals, states, actions)

    def freeze_encoders(self):
        with self.lock:
            self.bot.freeze_encoders()
    def unfreeze_encoders(self):
        with self.lock:
            self.bot.unfreeze_encoders()
