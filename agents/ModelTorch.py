import sys, os
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(sys.path[0], os.path.join("..", "models")))

###########################################################
# TODO : predict_future, not necessary to move to cpu ...
###########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import threading, math, os

#from linear_model import CriticNN
#from linear_model import ActorNN
from simple_model import CriticNN
from simple_model import ActorNN

#from gru_model import CriticQ as CriticNN
#from lstm_model import ActorNN
#from gru_model import CriticGRU as CriticNN
from gru_model import CriticNN
from gru_model import ActorNN

from utils.attention import SimulationAttention

import pickle
losses = []

class SoftUpdateNetwork:
    def soft_update(self, tau):
        if not tau:
            return

        for target_w, explorer_w in zip(self.target.parameters(), self.explorer.parameters()):
            target_w.data.copy_(
                target_w.data * (1. - tau) + explorer_w.data * tau)

        self.target.remove_noise()
        self.explorer.sample_noise()

    def get(self):
        return self.target.state_dict(), self.explorer.state_dict()

    def set(self, weights):
        self.target.load_state_dict(weights[0])
        self.explorer.load_state_dict(weights[1])

    def share_memory(self):
        self.target.share_memory()
        self.explorer.share_memory()

class ActorNetwork(SoftUpdateNetwork):
    def __init__(self, task, cfg):
        torch.set_default_tensor_type(cfg['tensor'])

        self.state_size = cfg['her_state_size'] + task.state_size() * cfg['history_count']
        self.sim_count = task.subtasks_count()

        self.cfg = cfg
        self.lock = threading.RLock()

        self.device = task.device()
        self.target = ActorNN(task, cfg).to(self.device)
        self.explorer = ActorNN(task, cfg).to(self.device)

        if self.cfg['load'] and os.path.exists(os.path.join(self.cfg['model_path'], 'actor_target')):
            self.target.load_state_dict(torch.load(os.path.join(self.cfg['model_path'], 'actor_target')))
            self.explorer.load_state_dict(torch.load(os.path.join(self.cfg['model_path'], 'actor_explorer')))

        self.soft_update(1.)

        self.attention = SimulationAttention(task, cfg) if self.cfg['attention_enabled'] else None

        self.opt = torch.optim.Adam(
                self.explorer.parameters() if not self.attention else list(self.explorer.parameters()) + list(self.attention.parameters()),
                cfg['lr_actor'])

    def fit(self, states, advantages, actions, tau):
        states = torch.DoubleTensor(states).to(self.device)
        actions = torch.DoubleTensor(actions).to(self.device)

        def local_optim():
            grads = advantages.view(-1, 1)
            if self.attention:
                grads = self.attention(grads, states, actions)

            pgd_loss = grads.mean() if self.cfg['pg_mean'] else grads.sum()

            # safety checks
            pgd_loss = -torch.clamp(pgd_loss, min=-self.cfg['loss_min'], max=self.cfg['loss_min'])
            if pgd_loss != pgd_loss: # is nan!
                return
            if pgd_loss.abs() < 1e-5:
                return

            # debug out
            if self.cfg['dbgout_train']:
                print("\n(*) train>>", states.shape, pgd_loss, tau)
            if self.cfg['loss_debug']:
                losses.append(pgd_loss.detach().cpu().numpy())
                with open('losses.pickle', 'wb') as l:
                    pickle.dump(losses, l)

            #proceed to learning
            self.opt.zero_grad()
            pgd_loss.backward()#retain_graph=True)

        with self.lock:
            self.opt.step(local_optim)
            self.soft_update(tau)

            if self.cfg['save']:
                torch.save(self.target.state_dict(), os.path.join(self.cfg['model_path'], 'actor_target'))
                torch.save(self.explorer.state_dict(), os.path.join(self.cfg['model_path'], 'actor_explorer'))

    def predict_present(self, states, history):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        with self.lock:
            dist = self.explorer(states, history)
            features = self.explorer.features
            return dist, features

    def predict_future(self, states, history):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        with self.lock:
            dist = self.target(states, history)
            features = self.target.features
            return dist, features

    @staticmethod
    def new(task, cfg):
        return ActorNetwork(task, cfg)

class CriticNetwork(SoftUpdateNetwork):
    def __init__(self, task, cfg, objective_id):
        torch.set_default_tensor_type(cfg['tensor'])

        self.cfg = cfg
        self.objective_id = objective_id

        self.device = task.device()
        self.target = CriticNN(task, cfg).to(self.device)
        self.explorer = CriticNN(task, cfg).to(self.device)

        self.soft_update(1.)

        self.opt = torch.optim.Adam(self.explorer.parameters(), cfg['lr_critic'])

        self.lock = threading.RLock()

        self.state_size = self.cfg['history_count'] * task.state_size() + cfg['her_state_size']

        if self.cfg['load'] and os.path.exists(os.path.join(self.cfg['model_path'], 'critic_target_%s'%self.objective_id)):
            self.target.load_state_dict(torch.load(os.path.join(self.cfg['model_path'], 'critic_target_%s'%self.objective_id)))
            self.explorer.load_state_dict(torch.load(os.path.join(self.cfg['model_path'], 'critic_explorer_%s'%self.objective_id)))


    def fit(self, states, actions, history, rewards, tau):
        # return
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        rewards = torch.DoubleTensor(torch.from_numpy(rewards)).to(self.device)
        actions = torch.DoubleTensor(torch.from_numpy(actions)).to(self.device)
        history = torch.DoubleTensor(torch.from_numpy(
            history.reshape(1, states.size(0), -1))).to(self.device)

        def optim():
            self.opt.zero_grad()
#            loss = F.smooth_l1_loss(
            loss = F.mse_loss(
                    self.explorer(states, actions, history), rewards)
            loss.backward()
        with self.lock:
            self.opt.step(optim)
            self.soft_update(tau)

            if self.cfg['save']:
                torch.save(self.target.state_dict(), os.path.join(self.cfg['model_path'], 'critic_target_%s'%self.objective_id))
                torch.save(self.explorer.state_dict(), os.path.join(self.cfg['model_path'], 'critic_explorer_%s'%self.objective_id))

    def predict_present(self, states, actions, history):
        states = torch.DoubleTensor(torch.from_numpy(
            states.reshape(-1, self.state_size))).to(self.device)
        history = history.view(1, states.size(0), -1).to(self.device)
        actions = actions.view(states.size(0), -1).to(self.device)

        with self.lock:
            return self.explorer.forward(states, actions, history)

    def predict_future(self, states, actions, history):
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
    def new(task, cfg, objective_id):
        return CriticNetwork(task, cfg, objective_id)
