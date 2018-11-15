import torch
from utils.bot import Bot

class BotProxy:
    def __init__(self, lock, cfg, bot, device):
        self.bot = bot
        self.cfg = cfg
        self.lock = lock
        self.device = device

        self.state_size = cfg['her_state_size'] + self.bot.state_size * cfg['history_count']

    def exploit(self, states, history):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        with self.lock: # dist in DQN should be similiar proxy like utils.policy.DDPGDist
            dist, features = self.bot.exploit(states, history)
            return (dist.sample().detach().cpu().numpy(),
                    features.detach().cpu().numpy())

    def explore(self, objective_id, states, history):
        ind = objective_id if self.cfg['detached_actor'] else 0
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        with self.lock:
            return self.bot.explore(ind, states, history)

    def qa_future(self, ind, states, history, actions):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        actions = torch.DoubleTensor(torch.from_numpy(actions)).to(self.device)

        actions = actions.view(states.size(0), -1)
        assert states.size(0) == actions.size(0)

        with self.lock:
            return self.bot.qa_future(ind, states, history, actions).detach().cpu().numpy()

    def qa_present(self, ind, states, history, actions):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = history.view(1, states.size(0), -1).to(self.device)
        actions = actions.view(states.size(0), -1).to(self.device)

        with self.lock:
            return self.bot.qa_present(ind, states, history, actions)

    def q_future(self, ind, states, history):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        with self.lock:
            return self.bot.q_future(ind, states, history).detach()

    def q_explore(self, objective_id, states, history):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        a_ind = objective_id if self.cfg['detached_actor'] else 0
        with self.lock:
            return self.bot.q_explore(a_ind, objective_id, states, history)

    def learn_actor(self, states, advantages, actions, tau):
        states = torch.DoubleTensor(states).to(self.device)
        actions = torch.DoubleTensor(actions).to(self.device)

        with self.lock:
            return self.bot.learn_actor(states, advantages, actions, tau)

    def learn_critic(self, ind, states, history, actions, qa_target, tau):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        actions = torch.DoubleTensor(torch.from_numpy(actions)).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        with self.lock:
            return self.bot.learn_critic(ind, states, history, actions, qa_target, tau)

    def alpha_sync(self, blacklist):
        self.bot.alpha_sync(blacklist)

    def reevaluate(self, objective_id, states, actions):
        ind = objective_id if self.cfg['detached_actor'] else 0
        states = torch.from_numpy(states)
        with self.lock:
            return self.bot.reevaluate(ind, states, actions)
