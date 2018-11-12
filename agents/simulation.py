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
        self.bot_id = bot_id
        self.objective_id = objective_id
        self.task = task_info.new(self.cfg, bot_id, self.objective_id)

        self.actor = shared_actor
#        self.master_actor = shared_actor
#        self.actor = Actor(model_actor.new(task_info, cfg))

        self.critic = model.new(
                task_info,
                self.task.device(),
                self.cfg,
                "%i_%i"%(bot_id, self.objective_id))
        self.critic.share_memory()

        self.best_max_step = self.cfg['max_n_step']
        self.delta_step = self.cfg['send_exp_delta']
        assert 0 == self.cfg['critic_learn_delta'] % self.delta_step, "send_exp_delta must be fraction of critic_learn_delta"

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
        self.critics = [Critic(cfg, self.critic, self.task, task_info,
            bot_id, self.objective_id, i,
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

# REFACTOR -> need proxy, as different critic schema is planned to future update
    def q_a_function(self, actions, probs, states, features, td_targets):
        dist, features = self.actor.get_action_w_grad(self.objective_id, states, features)

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
        if self.cfg['detach_eval']:
            self.stop = False
    #        looper = threading.Thread(target=self._eval_loop)
            looper = Process(target=self._eval_loop)
            looper.start()

        while True:
            seeds = self.keep_exploring.get()
            if None == seeds:
                break
            self._run(seeds)
            self.signal.put(True) # sync

        for c in self.done:
            c.put(True)

        if self.cfg['detach_eval']:
            self.stop = True
            looper.join()

    def _run(self, seeds):
        self.count += 1
        score_board = []

        exp_delta = self.delta_step + self.n_step

        for e, seed in enumerate(seeds):
            states = []
            actions = []
            probs = []
            features = []

            score = 0
            rewards = []
            goods = []

            state = self.task.reset(seed)[0]
            next_state = state

            f_pi = np.zeros(shape=(1, 1, self.cfg['history_features']))
            history = deque(maxlen=self.cfg['history_count'])
            for s in [np.zeros(len(state))] * self.cfg['history_count']:
                history.append(np.vstack(s))

            features += [f_pi] * 1

            last = -1
            done = False
            while True:
                if not self.cfg['detach_eval']:
                    self._single_eval()

                while self.cfg['postpone_exploring_while_learning'] and not self.td_backdoor.empty():
                    time.sleep(.1)

                state = next_state
                history.append(np.vstack(state))
                state = np.vstack(history).squeeze(1)
                state = self.task.transform_state(state) # move to AC-encoder .. TODO next changes

                norm_state = self.task.normalize_state(state.copy()) # move to AC-decoder .. TODO next changes
                dist, f_pi = self.actor.get_action_wo_grad(self.objective_id, norm_state, features[-1])
                a_pi = dist.sample().detach().cpu().numpy() # sample from distribution does not have gradient!

                if done:
                    break

                if len(rewards) >= self.max_n_episode:
                    break

                action, next_state, reward, done, good = self.task.step(a_pi)

                prob = dist.log_prob(torch.tensor(action)).detach().cpu().numpy()

                # here is action instead of a_pi on purpose ~ let user tell us what action he really take!
                actions.append(action)
                probs.append(prob)
                rewards.append(reward)
                states.append(state)
                goods.append(good)

                temp = self._share_experience(e, len(states), last)
                if temp != last:
                    self._do_fast_train(
                            states[-exp_delta:-self.n_step],
                            features[-exp_delta:-self.n_step],
                            actions[-exp_delta:-self.n_step],
                            probs[-exp_delta:-self.n_step],
                            rewards[-exp_delta:-self.n_step],
                            states[-self.delta_step:],
                            features[-self.delta_step:],
                            goods[-exp_delta:-self.n_step],
                            e, actions[-self.n_step])
                last = temp

                features.append(f_pi)

                score += reward * self.discount**len(rewards)

                self._print_stats(e, rewards, a_pi)

            self._do_last_train(
                    states[last:],
                    features[last:],
                    actions[last:],
                    probs[last:],
                    rewards[last:],
                    goods[last:],
                    state, e, a_pi)

            self.best_max_step = max(self.best_max_step, np.sign(self.cfg['max_n_step']) * len(rewards))

            self._print_stats(e, rewards, a_pi)

            if not self.cfg['detach_eval']:
                self._single_eval()

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

        if 0 != self.bot_id:
            return # info for main bot only

        if not self.cfg['dbgout']:
            print("\rstep:{:4d} :: {:.2f} [{}]".format(len(rewards), sum(rewards), self.count), end="")
        else:
            print("\r[{:d}>{:4d}::{:6d}] training = {:2d}, steps = {:3d}, max_step = {:3d}, reward={:2f} ::{}: {}".format(
                self.bot_id, self.count, self.task.iter_count(), e, len(rewards), abs(self.best_max_step), sum(rewards), a_pi, debug_out), end="")

    def _share_experience(self, e, total, last):
        delta = e + total
        if (delta - self.n_step) % self.delta_step:
            return last# dont overlap
        if total < self.n_step * 2:
            return last# not enough data
        if total < len(self.critics) + self.n_step:
            return last# not enough data
        return total - self.n_step

    def _do_fast_train(self, states, features, actions, probs, rewards, n_states, n_features, goods, e, action):
        self._forward(states, rewards, actions, probs, features, n_states, n_features, goods, e, action, False)

    def _do_last_train(self, states, features, actions, probs, rewards, goods, final_state, e, action):
        states += [final_state] * self.n_step
        features += [features[-1]] * (self.n_step - 1)

        self._forward(
            states[:-self.n_step],
            rewards,
            actions,
            probs,
            features[:-self.n_step],
            states[self.n_step:],
            features[self.n_step:],
            goods,
            e, action, True)

    def _forward(self, states, rewards, actions, probs, features, n_states, n_features, goods, delta, action, full):
        for i, exp_q in enumerate(self.exps):
            exp_q.put([full, delta, action, (
                states,
                rewards,
                actions,
                probs,
                features,
                n_states,
                n_features,
                goods,
                )])

    def _eval_loop(self):
        while not self.stop:
#            time.sleep(self.cfg["learn_loop_to"])
            time.sleep(.1)
            while all(not c.empty() for c in self.review):
                self._eval(self.td_backdoor)

# far less resource greedy, should not have per impact
    def _single_eval(self):
        if any(c.empty() for c in self.review):
            return
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
        f0 = np.vstack([f.reshape(f0[0].shape) for f in f0])
        n = np.vstack(n)
        fn = np.vstack([f.reshape(f0[0].shape) for f in fn])
        r = np.vstack(r)
#        td_gate.put([s, f0, r])
#        return

        if self.cfg['target_future_features']:
            an, fn = self.actor.predict(n, fn) # couple critic with target as well
        else:
            an, _ = self.actor.predict(n, fn) # make more sense, only feature of explorer

        def batch():
            yield (s, a0, f0.reshape(fn.shape), n, an, fn, r)

        with self.lock:
            # push for review of all critics
            for critic in self.comitee:
                critic.put(*batch())
            # get some reviewed data
            td_targets = [critic.get() for critic in self.td_gate] # arguments

        td_gate.put([a0, p0, s, f0, td_targets])
