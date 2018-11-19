import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.softnet import SoftUpdateNetwork
from utils.attention import SimulationAttention
from utils.ac import ActorCritic

class Bot(SoftUpdateNetwork):
    def __init__(self, cfg, actor_id, encoder, Actor, Critic, state_size, action_size, wrap_action, wrap_value):
        self.cfg = cfg
        self.actor_id = actor_id

        self.state_size = state_size

        self.ac_explorer = ActorCritic(
                encoder,
                [Actor(
                    encoder.total_size(), action_size, wrap_action, cfg
                    ) for i in range(1 if not cfg['detached_actor'] else cfg['n_simulations'])],
                [Critic(
                    encoder.total_size(), action_size, wrap_value, cfg
                    ) for i in range(cfg['n_simulations'])])

        self.attention = None if not self.cfg['attention_enabled'] else SimulationAttention(state_size, action_size, cfg)

        # set optimizers, expriment also with RMSprop, SGD w/ momentum
        self.c_opt = optim.Adam(self.ac_explorer.critic_parameters(), lr=self.cfg['lr_critic'])

        self.a_opt = optim.Adam(
                self.ac_explorer.actor_parameters() if not self.cfg['attention_enabled'] else np.concatenate([
                    list(self.attention.parameters()), self.ac_explorer.actor_parameters()]),
                lr=self.cfg['lr_actor'])

        self.ac_target = ActorCritic(
                encoder,
                [Actor(
                    encoder.total_size(), action_size, wrap_action, cfg
                    )],
                [Critic(
                    encoder.total_size(), action_size, wrap_value, cfg
                    ) for i in range(cfg['n_simulations'])])
        # sync
        self.sync_explorers(update_critics=True)

    def exploit(self, states, history):
        return self.ac_target.action(0, states, history)

    def explore(self, ind, states, history):
        return self.ac_explorer.action(ind, states, history)

    def qa_future(self, ind, states, history, actions):
        return self.ac_target.value(ind, states, history, actions)

    def qa_present(self, ind, states, history, actions):
        return self.ac_explorer.value(ind, states, history, actions)

    def q_future(self, ind, states, history):
        q, _ = self.ac_target(0, ind, states, history)
        return q

    def q_explore(self, a_ind, c_ind, states, history):
        return self.ac_explorer(a_ind, c_ind, states, history)

    def learn_actor(self, states, advantages, actions, tau):
        def local_optim():
            grads = advantages.view(-1, 1)
            if self.attention:
                grads = self.attention(grads, states, actions)

            pgd_loss = -(grads.mean() if self.cfg['pg_mean'] else grads.sum())

            print(">>train", pgd_loss, len(advantages))

            #proceed to learning
            self.a_opt.zero_grad()
#            nn.utils.clip_grad_norm_(self.ac_explorer.actor_parameters(), 1)

            pgd_loss.backward()#retain_graph=True)

#        params = [p.detach().clone() for p in list(self.ac_explorer.norm.parameters())]
        self.a_opt.step(local_optim)
#        assert not all(all(all(e==r) if e.dim() else e==r for e, r in zip(q, w)) for q, w in zip(params, list(self.ac_explorer.norm.parameters())))

        self.soft_mean_update(tau)
        self.save_models(self.cfg, self.actor_id, "actor")

    def learn_critic(self, ind, states, history, actions, qa_target, tau):
        def local_optim():
            self.c_opt.zero_grad()
#            loss = F.smooth_l1_loss(
            loss = F.mse_loss(
                    self.ac_explorer.value(ind, states, history, actions), qa_target)
            nn.utils.clip_grad_norm_(self.ac_explorer.critic[ind].parameters(), 1)
            loss.backward()

        self.c_opt.step(local_optim)
        self.soft_update(ind, tau)
# unlocked...
        self.save_models(self.cfg, self.actor_id, "critic")

    def alpha_sync(self, blacklist):
        self.load_models(self.cfg, 0, "actor", blacklist) # help apha actor to explore ~ multi agent coop !!
        self.sync_explorers() # load only alpha actor, keep critics our own

    def reevaluate(self, ind, states, actions):
        s, f = self.ac_explorer.extract_features(states)
        p = self.ac_explorer.actor[ind](s).log_prob(
                torch.tensor(actions)).detach().cpu().numpy()
        return f, p
