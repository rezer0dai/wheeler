import math, random, time, threading
import numpy as np

import torch
from torch.autograd import Variable
import torch.multiprocessing

from collections import deque
from utils.curiosity import CuriosityPrio

from baselines.common.schedules import LinearSchedule

class Critic(torch.multiprocessing.Process):
    def __init__(self, cfg, model, task, xid, stop, exps, review, comitee, grads, stats, complete, actor):
        super(Critic, self).__init__()

        self.stop = stop
        self.stats = stats
        self.exps = exps
        self.review = review
        self.comitee = comitee
        self.grads = grads
        self.complete = complete

        self.xid = xid
        self.cfg = cfg

        self.task = task
        self.fast_experience = []

        self.device = task.device()

        self.debug_out = "x" * 10
        self.debug_out_ex = "y" * 10

        self.n_step = self.cfg['n_step']
        self.discount = self.cfg['discount_rate']
        self.batch_size = self.cfg['batch_size']

        self.counter = 0
        self.tau = LinearSchedule(cfg['tau_replay_counter'],
               initial_p=self.cfg['tau_base'],
               final_p=cfg['tau_final'])

        self.lock = threading.RLock() # we protect only write operations ~ yeah yeah not so good :)
        self.replay = task.make_replay_buffer(cfg, actor)

        self.critic = model.new(task, self.cfg, "%i_%i"%(task.xid, xid))
        self.critic.share_memory()

        self.curiosity = CuriosityPrio(task, cfg)

    def q_a_function(self, states, actions, features, td_targets):
# even in multiprocess seems critic is shared from main process
        q_a = self.critic.predict_present(states, actions, features) # grads are active now

        if not self.cfg['advantages_enabled']:
            return -1. * q_a

        td_error = torch.DoubleTensor(td_targets).to(self.device) - q_a
        if not self.cfg['advantages_boost']:
            return td_error

        for i, e in enumerate(td_error):
            td_error[i] = e if abs(e) > 1e-3 else -q_a[i]
        return td_error

    def run(self):
        training = threading.Thread(target=self._training_loop)
        learning = threading.Thread(target=self._learning_loop)

        training.start()
        learning.start()

        self.stop.get()

        self.comitee.put(None)
        learning.join()

        self.exps.put(None)
        training.join()

    def _learning_loop(self):
        while True:
            batch = self.comitee.get()
            if None == batch:
                break
            grad = self._eval(batch)
            self.grads.put(grad)
            self.stats.put(self.debug_out_ex)

    def _training_loop(self):
        while True:
            exp = self.exps.get()
            if None == exp:
                break

            full, exp = exp
            if not full:
                self._inject(*exp)
            else:
                self._train(*exp)

            if not self.cfg['critic_learn_delta']:
                continue
            if not len(self.fast_experience):
                continue
            if self.fast_experience.shape[1] % self.cfg['critic_learn_delta']:
                continue

            batch = self._do_sampling()
            if None == batch:
                continue
            self.review.put(batch)

    def _inject(self, states, _, actions, features, n_states, n_rewards, __, n_features):
#        return
        _, _, _, s_norm, n_norm = self.stack_inputs(actions, states, n_states)
        fast_experience = np.vstack(
            map(lambda i: (
                s_norm[i],
                actions[i],
                features[i],
                n_norm[i],
                n_rewards[i],
                n_features[i]), range(len(states))))

        if not len(self.fast_experience):
            self.fast_experience = fast_experience.T
        else:
            self.fast_experience = np.vstack([
                self.fast_experience.T,
                fast_experience]).T

    def _train(self, states, rewards, actions, features, n_states, n_rewards, n_actions, n_features):
        if not len(states):
            return

        actions, states, n_states, s_norm, n_norm = self.stack_inputs(actions, states, n_states)

        self._update_memory(states, rewards, actions, features,
                n_states, n_rewards, n_features,
                s_norm, n_norm)

        self._update_normalizer(states)

        self._td_lambda()

        if not self.cfg['reinforce_clip']:
            return

# else we will do reinforce
        rewards = np.vstack(rewards)
        features = np.stack(features)

        self.critic.fit( # lets bias it towards real stuff ...
            s_norm,
            actions,
            features,
            rewards,
            self.cfg['tau_clip'])

    def _td_lambda(self):
        for _ in range(self.cfg['td_lambda_replay_count']):
            samples = self._select()
            if None == samples:
                break
            self.review.put(samples.T)
        self.complete.put(True)

    def _update_normalizer(self, states):
        self.task.update_normalizer(states[
                    random.sample(range(len(states)), random.randint(1, len(states)))])
        if len(self.replay) < self.batch_size:
            return None
        s, _, _, _, n, _, _ = self.replay.sample(self.batch_size)
        self.task.update_normalizer(s)
        self.task.update_normalizer(n)

    def _update_memory(self,
            states, rewards, actions, features,
            n_states, n_rewards, n_features,
            s_norm, n_norm):

        prios = self.curiosity.weight(s_norm, n_norm, actions)
        with self.lock:
            self.replay.add(
                map(lambda i: (
                    states[i],
                    rewards[i],
                    actions[i],
                    features[i],
                    n_states[i],
                    n_rewards[i],
                    n_features[i]), range(len(states))),
                prios)
        self.curiosity.update(s_norm, n_norm, actions)

    def _eval(self, args):
        states, actions, features, n_states, n_action, n_features, n_rewards = args

        assert len(n_features) == len(features), "features missmatch"
        if len(n_features) != len(features):
            return

        future = self.critic.predict_future(n_states, n_action, n_features)
        td_targets = n_rewards + (self.discount ** self.n_step) * future

        self.counter += 1
        self.critic.fit(states, actions, features, td_targets,
                self.tau.value(self.counter) * (0 == self.counter % self.cfg['critic_update_delay']))

        self.debug_out_ex = "[ TARGET:{:2f} replay::{} ]<----".format(
                td_targets[-1].item(), len(self.replay))

        return td_targets

    def _do_sampling(self):
        batch = self._fast_exp()
        if None == batch:
            return None

        samples = self._select()
        if None != samples:
            batch = np.vstack([batch, samples])

        return batch.T

    def _fast_exp(self):
        if not len(self.fast_experience):
            return None
        if len(self.replay) < self.batch_size and self.fast_experience.shape[1] < self.batch_size:
            return None
        batch = np.vstack(zip(*self.fast_experience))
        self.fast_experience = []
        return batch

    def _select(self):
        if len(self.replay) < self.batch_size:
            return None

        states, rewards, actions, features, n_states, n_rewards, n_features = self.replay.sample(self.batch_size)
        if not len(actions):
            return None

# ah uh, we would like to skip stacking, unnecessary perf overhead...
        actions, _, _, s_norm, n_norm = self.stack_inputs(actions, states, n_states)

        prios = self.curiosity.weight(s_norm, n_norm, actions)
        with self.lock:
            self.replay.update(prios)

        return np.vstack(
            map(lambda i: (
                s_norm[i],
                actions[i],
                features[i],
                n_norm[i],
                n_rewards[i],
                n_features[i]), range(len(states))))

    def stack_inputs(self, actions, states, n_states):
        a, s, n = (np.vstack(actions),
                np.vstack(states),
                np.vstack(n_states))
        s_norm = self.task.normalize_state(s.copy())
        n_norm = self.task.normalize_state(n.copy())
        return (a, s, n, s_norm, n_norm)
