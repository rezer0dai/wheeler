from __future__ import print_function
import time
import numpy as np
from collections import deque

import torch

from agents.critic import Critic

import threading
from concurrent.futures import ThreadPoolExecutor

class Simulation:
    def __init__(self, cfg, model, task, present_w_grads, present_wo_grads, future_wo_grads):
        self.cfg = cfg

        if not self.cfg['threading']:
            self.cfg['collector_scale_factor'] = 0

        # well not sure if i assign here actor itself if it will be copy or ref used later
        self.present_w_grads = present_w_grads
        self.present_wo_grads = present_wo_grads
        self.future_wo_grads = future_wo_grads

        self.batches = deque(maxlen=10)
        self.best_max_step = self.cfg['max_n_step']

        self.count = 0
        self.n_step = self.cfg['n_step']
        self.discount = self.cfg['discount_rate']
        self.max_n_episode = task.max_n_episode()

        self.batch_size = self.cfg['batch_size']
        self.critics = [Critic(cfg, model, task, i) for i in range(self.cfg['n_critics'])]

        self.lock = threading.RLock()

        self.stop = True
        self.xid = task.xid

    def get_grads(self):
        while not self.stop:
            with self.lock:
                if len(self.batches):
                    break
            time.sleep(self.cfg['exp_sampler_wait'])

        with self.lock:
            if len(self.batches):
                return self.batches.pop()#[-1]# the newest coolest ...
        return np.vstack([[[]] * 3])

    def turnon(self, task, n_episodes, actor):
        self.stop = False
        collectors = [threading.Thread(target=self.__eval_loop) for _ in range(self.cfg['collector_scale_factor'])]
        for collector in collectors:
            collector.start()

        score = self.__run_critics(task, n_episodes, actor)

        self.stop = True
        for collector in collectors:
            collector.join()

        return score

    def __run_critics(self, task, n_episodes, actor):
        for critic in self.critics:
            critic.turnon()
        score = self.__run_task(task, n_episodes, actor)
        for critic in self.critics:
            critic.shutdown()
        return score

    def __run_task(self, task, n_episodes, actor):
        return self.__run(task, n_episodes, actor)

    def __eval_loop(self):
        while not self.stop:
            self.__eval()
            time.sleep(self.cfg['exp_sampler_wait'])

    def __run(self, task, n_episodes, actor):
        self.count += 1
        score_board = []

        for e in range(n_episodes):
            states = []
            actions = []
            features = []

            score = 0
            rewards = []
            goods = []

            state = task.reset()

            # for i in range(len(state)):
                # state[i] = len(rewards)

            f_pi = np.zeros(self.cfg['action_features'])
            history = deque(maxlen=self.cfg['history_count'])
            for s in [state] * self.cfg['history_count']:
                history.append(np.vstack(s))

            while len(rewards) < self.max_n_episode:
                history.append(np.vstack(state))
                state = np.vstack(history).squeeze(1)#
                state = np.hstack([task.her_state(), state])

                #  print("\r  [ step", len(rewards), "]  ", end="")

                # if task.xid == 0: task.env.render()

                features.append(f_pi)
                while not self.cfg['threading'] and len(self.batches):
                    time.sleep(self.cfg['exp_sampler_wait'])

                a_pi, f_pi = self.present_wo_grads(state, f_pi)
                action, next_state, reward, done, good = task.step(a_pi)

                #  # as we do lots of transposing & stacking, better to double check
                #  # if in neural network we see really what we want ...
                # next_state = next_state.copy()
                #  for i in range(len(action)):
                #      action[i] = len(rewards)
                #  reward = len(rewards)
                #  for i in range(len(next_state)):
                #      next_state[i] = i * .2 + len(rewards)
                #  # double check this with NN
                #  print(action)
                #  print(next_state)
                #  print(state)
                #  print("~"*80)

                # here is action instead of a_pi on purpose ~ let user tell us what action he really take!
                actions.append(action)
                rewards.append(reward)
                states.append(state)
                goods.append(good)

                feats = f_pi

                self.__do_fast_train(
                        states[-2*self.n_step:],
                        features[-2*self.n_step:],
                        actions[-2*self.n_step:],
                        rewards[-2*self.n_step:],
                        goods[-2*self.n_step:],
                        e + len(states))

                if not self.cfg['threading'] and None != self.critics[0].batches:
                    self.__eval(actor)

                score += reward * self.discount**len(rewards)

                if done:
                    break
                state = next_state

#               if 0 == task.xid: print("\r[{:4d}::{:6d}] training = {:4d}, steps = {:2d}, max_step = {:2d}, reward={:2f} {}".format(
#                   self.count, task.iter_count(), e, len(rewards), abs(self.best_max_step), sum(rewards), self.critics[0].debug_out_ex), end="")

            history.append(np.vstack(next_state))
            next_state = np.hstack([task.her_state(), np.vstack(history).squeeze(1)])
            self.__do_full_ep_train(states, features, actions, rewards, goods, next_state, e)

            self.best_max_step = max(self.best_max_step, np.sign(self.cfg['max_n_step']) * len(rewards))
            score_board = np.hstack([score_board, len(rewards), rewards, self.best_max_step])

            print("\r[{:4d}::{:6d}] training = {:4d}, steps = {:2d}, max_step = {:2d}, accumulated {:4f}<->{:4f}, replay {:4d} name:<{}> {}".format(
                self.count, task.iter_count(), e, len(rewards), abs(self.best_max_step), score, sum(rewards), len(self.critics[0].replay), task.name(), self.critics[0].debug_out), end="")  # [debug]

        return score_board

    def __do_fast_train(self, states, features, actions, rewards, goods, delta):
        """
        advantage of A2C .. A3C .. methods are that they can learn online
         ~ we dont need to wait untill full episode
         ~ therefore we can adapt during applying some policy ad update it on the fly
        """
        # well this assertion does not particulary means that will be not effective to train otherwise
        # buuut, easy to do implementation currently like this :) and should be good trade-off "baseline" ~ maybe ...
        if delta % self.n_step:
            return # dont overlap ~ ensure critics has some unique data ...
        if len(states) < self.n_step * 2:
            return # not enough data
        if not sum(goods[self.n_step:]):
            return # nothing good to learn from this

        rewards += [0.] * self.n_step
        states, _, actions, features, n_states, n_rewards, n_actions, n_features = self.__process_data(
                states, actions, features, rewards, goods)

        max_ind = len(states)

        # print("fast ~>", len(states), max_ind, rewards, n_rewards)
        self.__inject_into_critic(
            states[:max_ind],
            rewards[:max_ind],
            actions[:max_ind],
            features[:max_ind],
            n_states[:max_ind],
            n_rewards[:max_ind],
            n_features[:max_ind],
            delta)

    def __do_full_ep_train(self, states, features, actions, rewards, goods, next_state, e):
        if not sum(goods):
            return

        states += [next_state] * self.n_step
        actions += [actions[-1]] * self.n_step
        features += [features[-1]] * self.n_step
        rewards += [0.] * self.n_step
        #filter only those from what we learned something ...
        goods += [False] * self.n_step

        states, rewards, actions, features, n_states, n_rewards, n_actions, n_features = self.__process_data(
                states, actions, features, rewards, goods)

        #we dont need to forward terminal state
        self.__train_critics(
            states[:-1],
            rewards[:-1],
            actions[:-1],
            features[:-1],
            n_states[:-1],
            n_rewards[:-1],
            n_actions[:-1],
            n_features[:-1],
            e)

    def __process_data(self, states, actions, features, rewards, goods):
            indicies = filter(lambda i: sum(goods[i:i+self.n_step*self.cfg['good_reach']]) > 0, range(len(goods) - self.n_step))

            return zip(*map(lambda i: [
                states[i],
                rewards[i],
                actions[i],
                features[i],
                states[i+self.n_step],
                [self.__n_reward(rewards[i:i+self.n_step], self.discount)],
                actions[i+self.n_step],
                features[i+self.n_step],
                ], indicies))

    def __resolve_index(self, max_step, ind):
        if not self.cfg['disjoint_critics'] or len(self.critics) > max_step:
            return 0, 1
        assert not self.cfg['disjoint_critics'] or len(self.critics) <= max_step, "can not disjoint critics if their # is higher than n_step!"
        return ind % len(self.critics), len(self.critics)

    def __inject_into_critic(self, states, rewards, actions, features, n_states, n_rewards, n_features, e):
        for i, critic in enumerate(self.critics):
            ind, max_step = self.__resolve_index(len(states), e + i)
            critic.inject(#fast operation, and we want to slow down step exec by little anyway :)
                states[ind:][::max_step],
                rewards[ind:][::max_step],
                actions[ind:][::max_step],
                features[ind:][::max_step],
                n_states[ind:][::max_step],
                n_rewards[ind:][::max_step],
                n_features[ind:][::max_step],
                )

    def __train_critics(self, states, rewards, actions, features, n_states, n_rewards, n_actions, n_features, e):
        for i, critic in enumerate(self.critics):
            ind, max_step = self.__resolve_index(len(states), e + i)
            threading.Thread(# heavy processing
                target=critic.train, args = (
                    states[ind:][::max_step],
                    rewards[ind:][::max_step],
                    actions[ind:][::max_step],
                    features[ind:][::max_step],
                    n_states[ind:][::max_step],
                    n_rewards[ind:][::max_step],
                    n_actions[ind:][::max_step],
                    n_features[ind:][::max_step],
                    )
                ).start()
            # collect those to some array may be good idea .. TODO ..
            # once we done with simulation ( training over )
            # we want be sure no thrads whatsover is running wildly ..

    def __eval(self, actor = None):
        s, a0, f1, n, r, fn = np.hstack(map(lambda c: c.batch(), self.critics))
        if not len(s):
            return

        # convert
        s = np.vstack(s)
        a0 = np.vstack(a0)
        f1 = np.vstack(f1)
        n = np.vstack(n)
        r = np.vstack(r)
        fn = np.vstack(fn)

        # for ddpg and gae; REPLAY!!
        a1, _ = self.present_w_grads(s, f1)
        an, _ = self.future_wo_grads(n, fn)

        def batch():
            yield (s, a0, a1, f1, n, r, an, fn)

#        advantages = [c.eval(*batch()) for c in self.critics]

        pool = ThreadPoolExecutor(len(self.critics))
        advantages = pool.map(Critic.eval, self.critics, # what
            [*batch()] * len(self.critics)) # arguments

        # i would like to make it more general not tighter to pytorch, but for now ok
        v = torch.stack([w for w in advantages]).mean(0)

        assert len(v) == len(s) and len(s) == len(a0), "s - v - a lengt missmatch.."

# a => forwarding actions just to let know to back-end what was really choosen
        if None != actor: # single thread ~ easier to debug
            actor.learn(s, v, a0, 1e-3)

        with self.lock:
            self.batches.append([ s, v, a0 ])

    def __n_reward(self, rewards, gamma):
        """
        discounted n-step reward, for advantage calculation
        """
        return sum(map(lambda t_r: t_r[1] * gamma**t_r[0], enumerate(rewards)))
