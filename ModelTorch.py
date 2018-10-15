import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import threading, math, os

#from archarxiv import CriticNN
#from archarxiv import ActorNN
from gru_model import CriticNN
from gru_model import ActorNN

class SoftUpdateNetwork:
    def soft_update(self, tau):
        """
        # get little from local~new, and most from stable~old
        basically copied from : https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
         ~ optimized copy-ing ( in comparsion to MedalKeras )
         ~ also more readable syntax ( than i have before ) "model*(1-tau) + local*tau" is somehow clean to understand :)
        """
        for target_w, explorer_w in zip(self.target.parameters(), self.explorer.parameters()):
            target_w.data.copy_(
                target_w.data * (1. - tau) + explorer_w.data * tau)

class AttentionNN(nn.Module):
    def __init__(self, task, cfg):
        super(AttentionNN, self).__init__()

        self.sims_count = task.subtasks_count()
        self.state_size = (cfg['her_state_size'] + task.state_size() * cfg['history_count'] + task.action_size())

        self.net = nn.Sequential(
                nn.Linear(self.state_size * self.sims_count, cfg['attention_hidden']),
                nn.Tanh(),
                nn.Linear(cfg['attention_hidden'], self.sims_count)
                )

    def forward(self, states, actions):
        inputs = torch.cat([ states, actions ], dim=1)
        inputs = torch.cat([i for i in inputs.view(self.sims_count, -1, self.state_size)], dim=1)
        flow = self.net(inputs)
        return F.softmax(flow, dim=1)

class ActorNetwork(SoftUpdateNetwork):
    def __init__(self, task, cfg):
        torch.set_default_tensor_type(cfg['tensor'])

        self.state_size = cfg['her_state_size'] + task.state_size() * cfg['history_count']
        self.sim_count = task.subtasks_count()

        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = task.action_high - task.action_low
        print("ACTION SPACE:", task.action_low, task.action_high)

        self.cfg = cfg
        self.lock = threading.RLock()

        self.device = task.device()
        self.target = ActorNN(task, cfg).to(self.device)
        self.explorer = ActorNN(task, cfg).to(self.device)

        if os.path.exists(os.path.join(self.cfg['model_path'], 'actor_target')):
            self.target.load_state_dict(torch.load(os.path.join(self.cfg['model_path'], 'actor_target')))
            self.explorer.load_state_dict(torch.load(os.path.join(self.cfg['model_path'], 'actor_explorer')))

        self.soft_update(1.)

        self.attention = AttentionNN(task, cfg) if self.cfg['attention_enabled'] else None

        self.opt = torch.optim.Adam(
                self.explorer.parameters() if not self.attention else list(self.explorer.parameters()) + list(self.attention.parameters()),
                cfg['lr_actor'])

    def reset(self):
        self.explorer.sample_noise()

    def fit(self, states, advantages, actions, tau = .1):
        states = torch.DoubleTensor(states).to(self.device)
        actions = torch.DoubleTensor(actions).to(self.device)

        def local_optim():
            grads = advantages.view(-1, 1)#Variable(torch.from_numpy(advantages)).to(self.device)
            #apply attention ~ we have stacked data from X different simulations ( reward functions )
            if self.attention:
                attention = self.cfg['attention_amplifier'] * self.attention(states, actions)
                print("\n ................ ATTENTION", attention)
                grads = (grads.view(attention.t().shape) * attention.t()).view(grads.shape)

            #  # debugging NN, if simulation outputs come in same way as we expect ...
            #  print(actions[:10])
            #  print(states[:10])
            #  print(advantages[:10])
            #  print("="*80)

            pgd_loss = -grads.sum() if not self.cfg['pg_mean'] else -grads.mean()
            print("\n(*) train>>", states.shape, tau, pgd_loss)
            # print(pgd_loss,-torch.mean(torch.cat(grads)), -torch.sum(torch.cat(grads)))

            #proceed to learning
            self.opt.zero_grad()
            pgd_loss.backward(retain_graph=True)

        with self.lock:
            self.opt.step(local_optim)
            self.soft_update(tau)
            self.target.remove_noise()

            torch.save(self.target.state_dict(), os.path.join(self.cfg['model_path'], 'actor_target'))
            torch.save(self.explorer.state_dict(), os.path.join(self.cfg['model_path'], 'actor_explorer'))

    def predict_present(self, states, history):
        # return np.array([0]).reshape(1, 1, 1)
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        with self.lock:
            actions = self.explorer(states, history)
            features = self.explorer.features
            return actions, features

    def predict_future(self, states, history):
        # return np.array([0]).reshape(1, 1, 1)
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        with self.lock:
            actions = self.target(states, history).detach().cpu().numpy()
            features = self.target.features
            return actions, features

    @staticmethod
    def new(task, cfg):
        return ActorNetwork(task, cfg)

class CriticNetwork(SoftUpdateNetwork):
    def __init__(self, task, cfg, xid):
        torch.set_default_tensor_type(cfg['tensor'])

        self.cfg = cfg
        self.xid = xid

        self.device = task.device()
        self.target = CriticNN(task, cfg).to(self.device)
        self.explorer = CriticNN(task, cfg).to(self.device)

        if os.path.exists(os.path.join(self.cfg['model_path'], 'critic_target_%s'%self.xid)):
            self.target.load_state_dict(torch.load(os.path.join(self.cfg['model_path'], 'critic_target_%s'%self.xid)))
            self.explorer.load_state_dict(torch.load(os.path.join(self.cfg['model_path'], 'critic_explorer_%s'%self.xid)))

        self.soft_update(1.)

        self.opt = torch.optim.Adam(self.explorer.parameters(), cfg['lr_critic'])

        self.lock = threading.RLock()

        self.state_size = self.cfg['history_count'] * task.state_size() + cfg['her_state_size']

    def fit(self, states, actions, history, rewards, tau = .1):
        # return
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        rewards = torch.DoubleTensor(torch.from_numpy(rewards)).to(self.device)
        actions = torch.DoubleTensor(torch.from_numpy(actions)).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        def optim():
            self.opt.zero_grad()
            loss = F.smooth_l1_loss(
                    self.explorer(states, actions, history), rewards)
            loss.backward()
        with self.lock:
            self.opt.step(optim)
            self.soft_update(tau)

            torch.save(self.target.state_dict(), os.path.join(self.cfg['model_path'], 'critic_target_%s'%self.xid))
            torch.save(self.explorer.state_dict(), os.path.join(self.cfg['model_path'], 'critic_explorer_%s'%self.xid))

    def predict_present(self, states, actions, history):
        # return np.array([0] * np.vstack(states).shape[0]).reshape(-1, 1)
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)
        actions = actions.view(states.size(0), -1)

        with self.lock:
            return self.explorer.forward(states, actions, history)

    def predict_future(self, states, actions, history):
        # return np.array([0] * np.vstack(states).shape[0]).reshape(-1, 1)
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        actions = torch.DoubleTensor(torch.from_numpy(actions)).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        actions = actions.view(states.size(0), -1)
        assert states.size(0) == actions.size(0)

        with self.lock:
            return self.target.forward(states, actions, history).detach().cpu().numpy()

    @staticmethod
    def new(task, cfg, xid):
        return CriticNetwork(task, cfg, xid)
