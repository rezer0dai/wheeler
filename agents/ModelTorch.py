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

    def _load_models(self, cfg, model_id, prefix):
        if not cfg['load']:
            return
        
        target = os.path.join(cfg['model_path'], '%s_target_%s'%(prefix, model_id))
        explorer = os.path.join(cfg['model_path'], '%s_explorer_%s'%(prefix, model_id))
        if not os.path.exists(target) or not os.path.exists(explorer):
            return

        self.target.load_state_dict(torch.load(target))
        self.explorer.load_state_dict(torch.load(explorer))

    def _save_models(self, cfg, model_id, prefix):
        if not cfg['save']:
            return
        target = os.path.join(cfg['model_path'], '%s_target_%s'%(prefix, model_id))
        explorer = os.path.join(cfg['model_path'], '%s_explorer_%s'%(prefix, model_id))

        torch.save(self.target.state_dict(), target)
        torch.save(self.explorer.state_dict(), explorer)


class ActorNetwork(SoftUpdateNetwork):
    def __init__(self, task_info, cfg, actor_id):
        torch.set_default_tensor_type(cfg['tensor'])

        self.state_size = cfg['her_state_size'] + task_info.state_size * cfg['history_count']

        self.cfg = cfg
        self.actor_id = actor_id
        self.lock = threading.RLock()

        self.device = "cpu" if torch.cuda.device_count() else self.cfg['device']
        self.target = ActorNN(
                task_info.state_size, task_info.action_size, task_info.wrap_action, 
                cfg).to(self.device)

        self.explorer = ActorNN(
                task_info.state_size, task_info.action_size, task_info.wrap_action, 
                cfg).to(self.device)

        self.soft_update(1.)

        self.attention = None if not self.cfg['attention_enabled'] else SimulationAttention(
                task_info.state_size, task_info.action_size, cfg)

        self.opt = torch.optim.Adam(
                self.explorer.parameters() if not self.attention else list(
                    self.explorer.parameters()) + list(self.attention.parameters()),
                cfg['lr_actor'])

        self._load_models(self.cfg, self.actor_id, "actor")

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

            self._save_models(self.cfg, self.actor_id, "actor")

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
    def new(task_info, cfg, actor_id):
        return ActorNetwork(task_info, cfg, actor_id)

class CriticNetwork(SoftUpdateNetwork):
    def __init__(self, task_info, device, cfg, critic_id):
        torch.set_default_tensor_type(cfg['tensor'])

        self.cfg = cfg
        self.critic_id = critic_id

        self.device = device
        self.target = CriticNN(
                task_info.state_size, task_info.action_size, task_info.wrap_value, 
                cfg).to(self.device)

        self.explorer = CriticNN(
                task_info.state_size, task_info.action_size, task_info.wrap_value, 
                cfg).to(self.device)

        self.soft_update(1.)

        self.opt = torch.optim.Adam(self.explorer.parameters(), cfg['lr_critic'])

        self.lock = threading.RLock()

        self.state_size = self.cfg['history_count'] * task_info.state_size + cfg['her_state_size']

        self._load_models(self.cfg, critic_id, "critic")

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

            self._save_models(self.cfg, self.critic_id, "critic")

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
    def new(task_info, device, cfg, critic_id):
        return CriticNetwork(task_info, device, cfg, critic_id)
