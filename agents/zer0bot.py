import numpy as np
import math, time, sys, random
from concurrent.futures import ThreadPoolExecutor, wait

from agents.actor import Actor
from agents.simulation import Simulation

import torch
from torch.autograd import Variable
import torch.multiprocessing
from torch.multiprocessing import Queue, SimpleQueue

from baselines.common.schedules import LinearSchedule

class Zer0Bot:
    def __init__(self, bot_id, cfg, task_info, model_actor, model_critic):
        self.cfg = cfg
        self.bot_id = bot_id

        self.n_step = self.cfg['n_step']

        self.counter = 1
        self.tau = LinearSchedule(cfg['tau_replay_counter'],
               initial_p=self.cfg['tau_base'],
               final_p=cfg['tau_final'])

        self._setup_actor(model_actor, task_info)
        self._setup_critics(model_critic, model_actor, task_info)

    def _setup_actor(self, model, task_info):
        self.actor = Actor(model.new(task_info, self.cfg, "%i"%self.bot_id), self.cfg)
        self.actor.share_memory()

    def _setup_critics(self, model_critic, model_actor, task_info):
        self.td_gate, self.mcts_timeout, self.signal = zip(*[(
            Queue(), SimpleQueue(), Queue()) for i in range(self.cfg['n_simulations'])])

        self.simulations = [Simulation(
                self.cfg, model_critic, task_info, self.bot_id, i + 1, self.actor, model_actor,
                self.td_gate[i], self.mcts_timeout[i], self.signal[i]
                ) for i in range(task_info.cfg['n_simulations'])]

    def act(self, state, history):# get exploitation action ( stable actor )
        a, history = self.actor.predict(state, history)
#        a, history = self.actor.get_action_wo_grad(state, history)
        return (a, history)

    def start(self):
        for c in self.simulations:
            c.turnon()
            c.start()

    def train(self):
        #  seed = [random.randint(0, self.cfg['mcts_random_cap'])] * self.cfg['mcts_rounds']
        for c in self.mcts_timeout: # maybe we want to * n_episodes ...
            c.put([random.randint(0, self.cfg['mcts_random_cap'])] * self.cfg['mcts_rounds'])

        while all(c.empty() for c in self.signal):
            self._train_worker()

        for s in self.signal:
            s.get()

    def _train_worker(self):
        time.sleep(.1)
#        time.sleep(self.cfg["learn_loop_to"])
        status = self._update_policy(self.tau.value(self.counter))
        if not status:
            return
        self.counter += 1

    def _get_grads(self, simulation, s_p_td_a):
        s, w, a = simulation.q_a_function(*s_p_td_a)
        return s, w, a

    def _update_policy(self, tau):
# here is potential dead-lock; not everytime every critic push same number of batches to review!!
# but that is rather problematic for attention mechanism ...
        if any(c.empty() for c in self.td_gate):
            return False

# hmm this trick with td_backdoor instead of td_gate[i] is rather cryptic .. TODO : refactor ..
# imho incosisten naming with zer0bot and simulation .. fix as well ..
        states, grads, actions = zip(*map(
            lambda c: self._get_grads(c, c.td_backdoor.get()), self.simulations))

        if self.cfg["attention_enabled"]:
            # ok we will scatter additional info which is hard to be weighted w/o additional info
            gran = min(map(len, grads))
            states = np.vstack([s[:gran] for s in states])
            actions = np.vstack([a[:gran] for a in actions])
            grads = torch.cat([g[:gran] for g in grads])
        else:
            states = np.vstack(states)
            actions = np.vstack(actions)
            gards = torch.cat(grads)

        tau = 1. if not self.cfg['ddpg'] else tau * (0 == self.counter % self.cfg['actor_update_delay'])
        self.actor.learn(states, gards, actions, tau)
        return True
