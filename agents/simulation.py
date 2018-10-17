from __future__ import print_function
import time, threading
import numpy as np
from collections import deque

from agents.actor import Actor
from agents.critic import Critic

import torch
import torch.multiprocessing
from torch.multiprocessing import Queue, SimpleQueue, Process

class Simulation(torch.multiprocessing.Process):
    def __init__(self, cfg, model, task, xid, shared_actor, model_actor, a_grads, keep_exploring, signal):
        super(Simulation, self).__init__()

        self.cfg = cfg
        self.xid = xid
        self.task = task.new(self.xid)

        self.actor = shared_actor
#        self.master_actor = shared_actor
#        self.actor = Actor(model_actor.new(task, cfg))

        self.best_max_step = self.cfg['max_n_step']

        self.count = 0
        self.n_step = self.cfg['n_step']
        self.discount = self.cfg['discount_rate']
        self.max_n_episode = task.max_n_episode()

        self.batch_size = self.cfg['batch_size']

        self.td_backdoor, self.keep_exploring, self.signal = a_grads, keep_exploring, signal

        self.done = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.complete = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.stats = [Queue() for _ in range(self.cfg['n_critics'])]
        self.exps = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.review = [Queue() for _ in range(self.cfg['n_critics'])]
        self.comitee = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.td_gate = [SimpleQueue() for _ in range(self.cfg['n_critics'])]
        self.critics = [Critic(cfg, model, self.task, i,
            self.done[i],
            self.exps[i],
            self.review[i],
            self.comitee[i],
            self.td_gate[i],
            self.stats[i],
            self.complete[i], self.actor) for i in range(self.cfg['n_critics'])]

        self.lock = threading.RLock()

    def turnon(self):
        for c in self.critics:
            c.start()

    def q_a_function(self, states, features, td_targets):
        actions, features = self.actor.get_action_w_grad(states, features)
        features = features.unsqueeze(0).transpose(0, 2)

        w = [c.q_a_function(
                states,
                actions.clone(),
                features.clone(),
                td) for c, td in zip(self.critics, td_targets)]

        return states, actions.detach().cpu().numpy(), w

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

            for _ in seeds:
                for c in self.complete:
                    c.get()

            while any(not c.empty() for c in self.review):
                pass

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

            features += [f_pi] * 1#self.cfg['history_count']

            done = False
            self.learned = False
            while len(rewards) < self.max_n_episode:
#                self._eval(self.td_backdoor)

                state = next_state
                history.append(np.vstack(state))
                state = np.vstack(history).squeeze(1)

                state = self.task.transform_state(state)
                if done:
                    break

                norm_state = self.task.normalize_state(state.copy())
                a_pi, f_pi = self.actor.get_action_wo_grad(norm_state, features[-1])
                action, next_state, reward, done, good = self.task.step(a_pi.copy())

                # here is action instead of a_pi on purpose ~ let user tell us what action he really take!
                actions.append(action)
                features.append(f_pi)
                rewards.append(reward)
                states.append(state)
                goods.append(good)

                self._do_fast_train(
                        states[-2*self.n_step:],
                        features[-2*self.n_step-1:],#self.cfg['history_count']:],
                        actions[-2*self.n_step:],
                        rewards[-2*self.n_step:],
                        goods[-2*self.n_step:],
                        e + len(states))

                score += reward * self.discount**len(rewards)

                self._print_stats(e, rewards, a_pi)

            self._do_full_ep_train(
                    states,
                    features[:-1],#-self.cfg['history_count']],
                    actions, rewards, goods, state, e)

            self.best_max_step = max(self.best_max_step, np.sign(self.cfg['max_n_step']) * len(rewards))

            self._print_stats(e, rewards, a_pi)

#this round does not contributed to actors training, redo!
            if not self.learned:
                self._run([seed + 1])

    def _print_stats(self, e, rewards, a_pi):
        debug_out = ""
        if not self.stats[0].empty():
            debug_out = self.stats[0].get()

        self.learned = len(debug_out) > 0

#        if 0 != self.xid:
#            return

        if not self.cfg['dbgout']:
            print("\rstep:{:4d} :: {} [{}]".format(len(rewards), sum(rewards), self.count), end="")
        else:
            print("\r[{:4d}::{:6d}] training = {:2d}, steps = {:3d}, max_step = {:3d}, reward={:2f} ::{}: {}".format(
                self.count, self.task.iter_count(), e, len(rewards), abs(self.best_max_step), sum(rewards), a_pi, debug_out), end="")

    def _do_fast_train(self, states, features, actions, rewards, goods, delta):
        if delta % self.n_step:
            return # dont overlap ~ ensure critics has some unique data ...
        if len(states) < self.n_step * 2:
            return # not enough data

        rewards += [0.] * self.n_step
        states, _, actions, features, n_states, n_rewards, n_actions, n_features = self._process_data(
                states, actions, features, rewards, [True]*len(goods))#goods)

        max_ind = len(states)

        # print("fast ~>", len(states), max_ind, rewards, n_rewards)
        self._forward(
            states[:max_ind],
            rewards[:max_ind],
            actions[:max_ind],
            features[:max_ind],
            n_states[:max_ind],
            n_rewards[:max_ind],
            n_actions[:max_ind],
            n_features[:max_ind],
            delta, False)

    def _do_full_ep_train(self, states, features, actions, rewards, goods, final_state, e):
        if not sum(goods):
            return

        states += [final_state] * self.n_step
        actions += [actions[-1]] * self.n_step
        features += [np.zeros(shape=features[0].shape)] * self.n_step
        rewards += [0.] * self.n_step
        #filter only those from what we learned something ...
        goods += [False] * self.n_step

        states, rewards, actions, features, n_states, n_rewards, n_actions, n_features = self._process_data(
                states, actions, features, rewards, goods)

        rewards = self._discount_rewards(rewards, self.discount)
        #we dont need to forward terminal state
        self._forward(
            states[:-1],
            rewards[:-1],
            actions[:-1],
            features[:-1],
            n_states[:-1],
            n_rewards[:-1],
            n_actions[:-1],
            n_features[:-1],
            e, True)

    def _process_data(self, states, actions, features, rewards, goods):
        indicies = filter(lambda i: sum(goods[i:i+self.n_step*self.cfg['good_reach']]) > 0, range(len(goods) - self.n_step))

        return zip(*map(lambda i: [
            states[i],
            rewards[i],
            actions[i],
            features[i],
            states[i+self.n_step],
            [self._n_reward(rewards[i:i+self.n_step], self.discount)],
            actions[i+self.n_step],
            features[i+self.n_step],
            ], indicies))

    def _resolve_index(self, max_step, ind):
        if not self.cfg['disjoint_critics'] or len(self.critics) > max_step:
            return 0, 1
        assert not self.cfg['disjoint_critics'] or len(self.critics) <= max_step, "can not disjoint critics if their # is higher than n_step!"
        return ind % len(self.critics), len(self.critics)

    def _forward(self, states, rewards, actions, features, n_states, n_rewards, n_actions, n_features, e, full):
        for i, exp_q in enumerate(self.exps):
            ind, max_step = self._resolve_index(len(states), e + i)
            exp_q.put([full, (
                states[ind:][::max_step],
                rewards[ind:][::max_step],
                actions[ind:][::max_step],
                features[ind:][::max_step],
                n_states[ind:][::max_step],
                n_rewards[ind:][::max_step],
                n_actions[ind:][::max_step],
                n_features[ind:][::max_step],
                )])

    def _n_reward(self, rewards, gamma):
        """
        discounted n-step reward, for advantage calculation
        """
        return sum(map(lambda t_r: t_r[1] * gamma**t_r[0], enumerate(rewards)))

    def _discount_rewards(self, r, gamma):
        """
        msft reinforcement learning explained course code, GAE approach
        """
        discounted_r = np.zeros(len(r))
        running_add = 0.
        for t in reversed(range(0, len(r))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def _eval_loop(self):
        while not self.stop:
#            time.sleep(self.cfg["learn_loop_to"])
            self._eval(self.td_backdoor)

    def _eval(self, td_gate):
#        if not td_gate.empty(): # no need to ask this, and in the end dead-lock issue
#            return
        if any(c.empty() for c in self.review):
            return

# ok lets examine updated actor, we doing off-policy anyway
#        self.actor.reload(self.master_actor)
        def flush(c):
            while not c.empty():
                yield c.get()
                break

#zip(*map(lambda c: zip(*flush(c)), self.review))#
# OUCH again we doing numpy stack magic ... TODO : refactor
        s, a0, f0, n, r, fn = np.hstack(map(lambda c: np.hstack(flush(c)), self.review))
        if not len(s):
            return

#        threading.Thread(target=self._replay, args=(td_gate, s, a0, f0, n, r, fn, )).start()

#    def _replay(self, td_gate, s, a0, f0, n, r, fn):
        s = np.vstack(s)
        a0 = np.vstack(a0)
        f0 = np.stack(f0)
        n = np.vstack(n)
        fn = np.stack(fn)
        r = np.vstack(r)
#        td_gate.put([s, f0, r])
#        return

# for ddpg and gae; REPLAY!! -> f1 <- target == more stable, explorer more dynamic ...
        an, fn = self.actor.predict(n, fn)
        _, f1 = self.actor.get_action_wo_grad(s, f0)#self.actor.predict(s, f0)#

        def batch():
            yield (s, a0, f1, n, an, fn, r)

        with self.lock:
    # push for review of all critics
            for critic in self.comitee:
                critic.put(*batch())
    # get some reviewed data
            td_targets = [critic.get() for critic in self.td_gate] # arguments

        td_gate.put([s, f0, td_targets])
