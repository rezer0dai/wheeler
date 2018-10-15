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
    def __init__(self, cfg, task, model_actor, model_critic):
        self.cfg = cfg

        self.task_main = task
        self.n_step = self.cfg['n_step']

        self.counter = 1
        self.tau = LinearSchedule(cfg['tau_replay_counter'],
               initial_p=self.cfg['tau_base'],
               final_p=cfg['tau_final'])

        self._setup_actor(model_actor)
        self._setup_critics(model_critic, model_actor)

    def _setup_actor(self, model):
        self.actor = Actor(model.new(self.task_main, self.cfg))
        self.actor.share_memory()

    def _setup_critics(self, model, model_actor):
        self.td_gate, self.mcts_timeout, self.signal, self.drop = zip(*[(
            Queue(), SimpleQueue(), Queue(), Queue()) for i in range(self.task_main.subtasks_count())])

        self.simulations = [Simulation(
                self.cfg, model, self.task_main, i, self.actor, model_actor,
                self.td_gate[i], self.mcts_timeout[i], self.signal[i], self.drop[i]
                ) for i in range(self.task_main.subtasks_count())]

    def act(self, state, history):# get exploitation action ( stable actor )
        a, history = self.actor.predict(state, history)
#        a, history = self.actor.get_action_wo_grad(state, history)
        return (a, history)

    def start(self):
        for c in self.simulations:
            c.turnon()
            c.start()

    def train(self, n_episodes):
        #  seed = [random.randint(0, self.cfg['mcts_random_cap'])] * self.cfg['mcts_rounds']
        for c in self.mcts_timeout: # maybe we want to * n_episodes ...
            c.put([random.randint(0, self.cfg['mcts_random_cap'])] * self.cfg['mcts_rounds'])

        while all(c.empty() for c in self.signal):
            self._train_worker()

        for d, s in zip(self.drop, self.signal):
            d.put(True) # fixing dead-lock
            s.get()

    def _train_worker(self):
#        time.sleep(self.cfg["learn_loop_to"])
        status = self._update_policy(self.tau.value(self.counter))
        if not status:
            return
        self.counter += 1
        self.actor.reset()

    def _get_grads(self, simulation, s_a_td):
        s, a, w = simulation.q_a_function(*s_a_td)
        return s, a, torch.stack(w).mean(0) # this is problematic if we want to use keras ?

    def _update_policy(self, tau):
# here is potential dead-lock; not everytime every critic push same number of batches to review!!
# but that is rather problematic for attention mechanism ...
        if any(c.empty() for c in self.td_gate):
            return False

# hmm this trick with td_backdoor instead of td_gate[i] is rather cryptic .. TODO : refactor ..
# imho incosisten naming with zer0bot and simulation .. fix as well ..
        states, actions, advantages = zip(*map(
            lambda c: self._get_grads(c, c.td_backdoor.get()), self.simulations))

        if self.cfg["attention_enabled"]:
            # ok we will scatter additional info which is hard to be weighted w/o additional info
            gran = min(map(len, advantages))
            states = np.vstack([s[:gran] for s in states])
            actions = np.vstack([a[:gran] for a in actions])
            advantages = torch.stack([v[:gran] for v in advantages])
        else:
            states = np.vstack(states)
            actions = np.vstack(actions)
            advantages = torch.cat(advantages)

        advantages = self._normalize(advantages)
        self.actor.learn(states, advantages, actions,
                tau * (0 == self.counter % self.cfg['actor_update_delay']))
        return True

    def _normalize(self, advantages):
        """
        work over standard mean, to avoid unecessary chaos in policy, source from OpenAI
        """
        if not self.cfg['normalize_advantages']:
            return advantages
        normalize = lambda a: (a - a.mean()) / a.std()
        return normalize(advantages)
