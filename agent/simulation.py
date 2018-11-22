from __future__ import print_function
import time
import numpy as np
from collections import deque

import torch
from torch.multiprocessing import SimpleQueue, Queue, Process

from agent.critic import critic_launch

from threading import Thread

def simulation_launch(cfg, bot, bot_id, objective_id, task_factory, loss, mcts, signal):
    ping, sync, share = SimpleQueue(), SimpleQueue(), Queue()
    stats = Queue() if not bot_id else None

    sim = Simulation(cfg, bot, bot_id, objective_id, task_factory)

    critic = Thread(#Process(#
            target=critic_launch,
            args=(cfg, bot, objective_id, task_factory, sim.task.update_goal, ping, sync, loss, share, stats,))
    critic.start()

    sim.explore(ping, sync, loss, mcts, signal, share, stats)
    print("SIMULATION OVER")
    critic.join()

    # TODO : ping scoping refactor, following slush refactor, dtor refactor ( sim + crit )
    for q in [ping, sync, share, stats]:
        while q is not None and not q.empty():
            q.get()
    # seems queues has still something to process .. check more python multiproc to avoid this timer ...
    time.sleep(3)#10

class Simulation:
    def __init__(self, cfg, bot, bot_id, objective_id, task_factory):
        self.cfg = cfg
        self.bot = bot
        self.bot_id = bot_id
        self.objective_id = objective_id

        self.stop = False

        self.task_factory = task_factory
        self.task = task_factory.new(cfg, bot_id, objective_id)

        self.best_max_step = self.cfg['max_n_step']
        self.delta_step = self.cfg['send_exp_delta']
        assert 0 == self.cfg['critic_learn_delta'] % self.delta_step, "send_exp_delta must be fraction of critic_learn_delta"

        self.count = 0
        self.n_step = self.cfg['n_step']
        self.max_n_episode = self.cfg['max_n_episode']

    def explore(self, ping, sync, loss_gate, mcts, signal, share_gate, stats):
        while True:
            seeds = mcts.get()
            if None == seeds:
                break
            scores = self._run(seeds, ping, share_gate, loss_gate, stats)
            signal.put(scores) # sync

        self._dtor(sync, share_gate)

    def _dtor(self, sync, share_gate):
        self.stop = True
        share_gate.put(None)
        sync.get()

    def _run(self, seeds, ping, share_gate, loss_gate, stats):
        self.count += 1
        exp_delta = self.delta_step + self.n_step

        scores = []
        for e, seed in enumerate(seeds):
            goals = []
            states = []
            actions = []
            probs = []
            features = []

            scores.append(0)
            rewards = []
            goods = []

            state = self.task.reset(seed)[0]
            next_state = state

            f_pi = np.zeros(shape=(1, 1, self.cfg['history_features']))
            history = deque(maxlen=self.cfg['history_count'])
            for s in [np.zeros(len(state))] * self.cfg['history_count']:
                history.append(np.vstack(s))

            features += [f_pi] * 1

            last = 0
            done = False
            frozen = False
            while True:
                frozen = self._sync(frozen, ping, loss_gate)

                state = next_state
                history.append(np.vstack(state))

                state = np.vstack(history).squeeze(1)
                goal = self.task.goal(self.objective_id)

                dist, f_pi = self.bot.explore(self.objective_id, goal, state, features[-1])

                a_pi = dist.sample().detach().cpu().numpy()
                f_pi = f_pi.detach().cpu().numpy()

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
                goals.append(goal)
                states.append(state)
                goods.append(good)

                temp = self._share_experience(e, len(states), last)
                if temp != last:
                    self._do_fast_train(share_gate,
                            goals[-exp_delta:-self.n_step],
                            states[-exp_delta:-self.n_step],
                            features[-exp_delta:-self.n_step],
                            actions[-exp_delta:-self.n_step],
                            probs[-exp_delta:-self.n_step],
                            rewards[-exp_delta:-self.n_step],
                            goals[-self.delta_step:],
                            states[-self.delta_step:],
                            features[-self.delta_step:],
                            goods[-exp_delta:-self.n_step],
                            actions[-self.n_step])
                last = temp

                features.append(f_pi)

                scores[-1] += reward

                self._print_stats(e, rewards, a_pi, stats)

            self._do_last_train(share_gate,
                    goals[last:],
                    states[last:],
                    features[last:],
                    actions[last:],
                    probs[last:],
                    rewards[last:],
                    goods[last:],
                    state, goal, a_pi)

            self.best_max_step = max(self.best_max_step, np.sign(self.cfg['max_n_step']) * len(rewards))

            self._print_stats(e, rewards, a_pi, stats)

# we must ensure each round at least one training happaned!
            if len(rewards) < self.n_step * 2:
                self._run([seed + 1], share_gate, loss_gate, stats)

        while self._sync(False, ping, loss_gate):
            pass # flush learning queue per train-round!!

        return scores

    def _print_stats(self, e, rewards, a_pi, stats):
        if 0 != self.bot_id:
            return # info for main bot only

        if not self.cfg['dbgout']:
            return print("\rstep:{:4d} :: {:.2f} [{}]".format(len(rewards), sum(rewards), self.count), end="")

        debug_out = ""
        while not stats.empty():
            debug_out = stats.get()

        print("\r[{:d}>{:4d}::{:6d}] training = {:2d}, steps = {:3d}, max_step = {:3d}, reward={:2f} ::{}: {}".format(
            self.bot_id, self.count, self.task.iter_count(), e, len(rewards), abs(self.best_max_step), sum(rewards), a_pi, debug_out), end="")

    def _share_experience(self, e, total, last):
        delta = e + total
        if (delta - self.n_step) % self.delta_step:
            return last# dont overlap
        if total < self.n_step * 2:
            return last# not enough data
        return total - self.n_step

    def _do_fast_train(self, shared_gate,
            goals, states, features, actions, probs, rewards,
            n_goals, n_states, n_features,
            goods, action):

        self._forward(shared_gate,
                goals, states, features, actions, probs, rewards,
                n_goals, n_states, n_features, goods,
                action, False)

    def _do_last_train(self, shared_gate,
            goals, states, features, actions, probs, rewards,
            goods, final_state, final_goal, action):

        goals += [final_goal] * self.n_step
        states += [final_state] * self.n_step
        features += [features[-1]] * (self.n_step - 1)

        self._forward(shared_gate,
            goals[:-self.n_step], states[:-self.n_step], features[:-self.n_step], actions, probs, rewards,
            goals[self.n_step:], states[self.n_step:], features[self.n_step:], goods,
            action, True)

    def _forward(self, shared_gate, goals, states, features, actions, probs, rewards, n_goals, n_states, n_features, goods, action, full):
        if self.stop:
            return
        shared_gate.put([full, action, (
            goals, states, features, actions, probs, rewards, n_goals, n_states, n_features, goods,
            )])

    def _sync(self, frozen, ping, loss_gate):
        for i in range(10 * self.cfg['postpone_exploring_while_learning']):
            if frozen and i > 4:
                break # dont wait so long everytime if it is overfilled ~ buggy prone then ...
            if ping.empty() and loss_gate.empty():
                break
            time.sleep(.1)
        return not ping.empty() or not loss_gate.empty()
