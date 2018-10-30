from __future__ import print_function
import time, threading
import numpy as np
from collections import deque

from agents.actor import Actor
from agents.critic import Critic

import torch
import torch.multiprocessing
from torch.multiprocessing import Queue, SimpleQueue, Process

from utils import policy

class Simulation(torch.multiprocessing.Process):
    def __init__(self, 
            cfg, model, task_info, 
            bot_id, objective_id, 
            shared_actor, model_actor, 
            a_grads, keep_exploring, signal):
        
        super(Simulation, self).__init__()

        self.cfg = cfg
        self.objective_id = objective_id
        self.task = task_info.new(self.objective_id, bot_id)

        self.actor = shared_actor
#        self.master_actor = shared_actor
#        self.actor = Actor(model_actor.new(task_info, cfg))

        self.best_max_step = self.cfg['max_n_step']
        self.delta_step = self.cfg['critic_learn_delta']

        self.count = 0
        self.n_step = self.cfg['n_step']
        self.discount = self.cfg['discount_rate']
        self.max_n_episode = self.cfg['max_n_episode']

        self.batch_size = self.cfg['batch_size']

        self.td_backdoor, self.keep_exploring, self.signal = a_grads, keep_exploring, signal

        self.done = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.complete = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.stats = [Queue() for _ in range(self.cfg['n_critics'])]
        self.exps = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.review = [Queue() for _ in range(self.cfg['n_critics'])]
        self.comitee = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.td_gate = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.critics = [Critic(cfg, model, self.task, task_info, bot_id, i + 1,
            self.done[i],
            self.exps[i],
            self.review[i],
            self.comitee[i],
            self.td_gate[i],
            self.stats[i],
            self.complete[i],
            self.actor) for i in range(self.cfg['n_critics'])]

        self.lock = threading.RLock()

    def turnon(self):
        for c in self.critics:
            c.start()

    def q_a_function(self, actions, probs, states, features, td_targets):
        dist, features = self.actor.get_action_w_grad(states, features)

        loss = [c.q_a_function(
                states,
                dist.sample(),
                features,
                td) for c, td in zip(self.critics, td_targets)]

        loss = torch.stack(loss).mean(0)
        if self.cfg['normalize_advantages']:
            loss = policy.normalize(loss)

        probs = torch.tensor(probs)
        new_probs = dist.log_prob(torch.tensor(actions))

        grads = policy.policy_loss(probs, new_probs, loss, self.cfg['ppo_eps'], self.cfg['dbgout_ratio'])

        return states, grads, actions

    def run(self):
        self.stop = False
#        looper = threading.Thread(target=self._eval_loop)
        looper = Process(target=self._eval_loop)#, args=(self.review, self.comitee, self.td_backdoor))
        looper.start()

        while True:
            seeds = self.keep_exploring.get()
            if None == seeds:
                break
            self._run(seeds)
            self.signal.put(True) # sync

        for c in self.done:
            c.put(True)

        self.stop = True
        looper.join()

    def _run(self, seeds):
        self.count += 1
        score_board = []

        for e, seed in enumerate(seeds):
            states = []
            actions = []
            probs = []
            features = []

            score = 0
            rewards = []
            goods = []

            state = self.task.reset(seed)
            next_state = state

            f_pi = np.zeros(shape=(1, 1, self.cfg['history_features']))
            history = deque(maxlen=self.cfg['history_count'])
            for s in [np.zeros(len(state))] * self.cfg['history_count']:
                history.append(np.vstack(s))

            features += [f_pi] * 1

            done = False
            while len(rewards) < self.max_n_episode:
#                self._eval(self.td_backdoor)

                state = next_state
                history.append(np.vstack(state))
                state = np.vstack(history).squeeze(1)

                state = self.task.transform_state(state)
                if done:
                    break

                norm_state = self.task.normalize_state(state.copy())
                dist, f_pi = self.actor.get_action_wo_grad(norm_state, features[-1])

                a_pi = dist.sample().detach().cpu().numpy() # sample from distribution does not have gradient!
                action, next_state, reward, done, good = self.task.step(a_pi)

                prob = dist.log_prob(torch.tensor(action)).detach().cpu().numpy()

                # here is action instead of a_pi on purpose ~ let user tell us what action he really take!
                actions.append(action)
                probs.append(prob)
                rewards.append(reward)
                states.append(state)
                goods.append(good)

                exp_delta = self.delta_step + self.n_step
                self._do_fast_train(
                        states[-exp_delta:],
                        features[-exp_delta:],
                        actions[-exp_delta:],
                        probs[-exp_delta:],
                        rewards[-exp_delta:],
                        goods[-exp_delta:],
                        e + len(states))

                features.append(f_pi)

                score += reward * self.discount**len(rewards)

                self._print_stats(e, rewards, a_pi)

            self._do_full_ep_train(
                    states,
                    features,
                    actions, probs, rewards, goods, state, e)

            self.best_max_step = max(self.best_max_step, np.sign(self.cfg['max_n_step']) * len(rewards))

            self._print_stats(e, rewards, a_pi)

            if any(c.empty() for c in self.complete):
#this round does not contributed to actors training, redo!
                self._run([seed + 1])

            for c in self.complete:
                while not c.empty():
                    c.get()

    def _print_stats(self, e, rewards, a_pi):
        debug_out = ""
        while not self.stats[0].empty():
            debug_out = self.stats[0].get()

        if 1 != self.objective_id:
            return

        if not self.cfg['dbgout']:
            print("\rstep:{:4d} :: {} [{}]".format(len(rewards), sum(rewards), self.count), end="")
        else:
            print("\r[{:4d}::{:6d}] training = {:2d}, steps = {:3d}, max_step = {:3d}, reward={:2f} ::{}: {}".format(
                self.count, self.task.iter_count(), e, len(rewards), abs(self.best_max_step), sum(rewards), a_pi, debug_out), end="")

    def _do_fast_train(self, states, features, actions, probs, rewards, goods, delta):
        if (delta - self.n_step) % self.delta_step:
            return # dont overlap
        if len(states) < self.n_step * 2:
            return # not enough data
        if len(states) < len(self.critics) + self.n_step:
            return # not enough data

        action = actions[-1]
        rewards += [0.] * self.n_step
        states, rewards, actions, probs, features, n_states, n_features = self._process_data(
                states, actions, probs, features, rewards, range(len(states) - self.n_step))
        # in fast experience we collect all signals, to buffer we can select based on good info ~
        # aka select important ones for future learning

        max_ind = len(states) - self.n_step

        self._forward(
            states[:max_ind],
            rewards[:max_ind],
            actions[:max_ind],
            probs[:max_ind],
            features[:max_ind],
            n_states[:max_ind],
            n_features[:max_ind],
            action, False)

    def _do_full_ep_train(self, states, features, actions, probs, rewards, goods, final_state, e):
        if len(states) < self.n_step:
            return # not enough data
        if not sum(goods):
            return

        states += [final_state] * self.n_step
        actions += [actions[-1]] * self.n_step
        features += [features[-1]] * (self.n_step - 1)
        rewards += [0.] * self.n_step

        #filter only those from what we learned something ...
        goods += [False] * self.n_step
        indicies = filter(
                lambda i: sum(goods[i:i+self.n_step*self.cfg['good_reach']]) > 0, range(len(goods) - self.n_step))

        states, rewards, actions, probs, features, n_states, n_features = self._process_data(
                states, actions, probs, features, rewards, indicies)

        #we dont need to forward terminal state
        self._forward(
            states[:-self.n_step],
            rewards[:-self.n_step],
            actions[:-self.n_step],
            probs[:-self.n_step],
            features[:-self.n_step],
            n_states[:-self.n_step],
            n_features[:-self.n_step],
            e, True)

    def _process_data(self, states, actions, probs, features, rewards, indicies):
        return zip(*map(lambda i: [
            states[i],
            rewards[i],
            actions[i],
            probs[i],
            features[i],
            states[i+self.n_step],
            features[i+self.n_step],
            ], indicies))

    def _forward(self, states, rewards, actions, probs, features, n_states, n_features, info, full):
        for i, exp_q in enumerate(self.exps):
            exp_q.put([full, info, (
                states,
                rewards,
                actions,
                probs,
                features,
                n_states,
                n_features,
                )])

    def _eval_loop(self):
        while not self.stop:
#            time.sleep(self.cfg["learn_loop_to"])
            time.sleep(.1)
            while all(not c.empty() for c in self.review):
                self._eval(self.td_backdoor)

    def _eval(self, td_gate):
# ok lets examine updated actor, we doing off-policy anyway
#        self.actor.reload(self.master_actor)
        def flush(c):
            while not c.empty():
                yield c.get()
                break

#zip(*map(lambda c: zip(*flush(c)), self.review))#
# OUCH again we doing numpy stack magic ... TODO : refactor
        s, a0, p0, f0, n, r, fn = np.hstack(map(lambda c: np.hstack(flush(c)), self.review))
        if not len(s):
            return

        threading.Thread(target=self._replay, args=(td_gate, s, a0, p0, f0, n, r, fn, )).start()
    def _replay(self, td_gate, s, a0, p0, f0, n, r, fn):

        s = np.vstack(s)
        a0 = np.vstack(a0)
        p0 = np.vstack(p0)
        f0 = np.array([f.reshape(f0[0].shape) for f in f0])
        n = np.vstack(n)
        fn = np.array([f.reshape(fn[0].shape) for f in fn])
        r = np.vstack(r)
#        td_gate.put([s, f0, r])
#        return

#        an, fn = self.actor.predict(n, fn)
        an, _ = self.actor.predict(n, fn)

        def batch():
            yield (s, a0, f0.reshape(fn.shape), n, an, fn, r)

        with self.lock:
            # push for review of all critics
            for critic in self.comitee:
                critic.put(*batch())
            # get some reviewed data
            td_targets = [critic.get() for critic in self.td_gate] # arguments

        td_gate.put([a0, p0, s, f0, td_targets])
