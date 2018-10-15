import math, random, time, threading
import numpy as np

from agents.replay import ReplayBuffer
from agents.curiosity import CuriosityPrio

import torch
from torch.autograd import Variable

from collections import deque

class Critic:
    def __init__(self, cfg, model, task, xid):
        """
        we want to be our replay buffer up to day ( +- ) with policy
            ~ if too big buffer, those rewards should be far away from reward which should current policy gain
            ~ however need to handle forgeting then :) TODO ...
                ~ i do believe it should be more about states than rewards

        + we want to be our network independent of :
            ~ this algorithm
            ~ framework used { tf, keras, pytorch, cntk, custom(zer0nn, ..), .. }
        """
        self.cfg = cfg

        self.lock = threading.RLock()

        self.device = task.device()

        self.lock_batch = threading.RLock()
        self.batches = None#[]#deque(maxlen=10)
        self.fast_experience = []

        self.debug_out = "x" * 10
        self.debug_out_ex = "x" * 10

        self.n_step = self.cfg['n_step']
        self.discount = self.cfg['discount_rate']
        self.batch_size = self.cfg['batch_size']
        print("REPLAY BUFF SIZE : ", self.batch_size * task.max_n_episode() // self.n_step)

        self.reward_scale = max(abs(self.cfg['max_reward_val']), abs(self.cfg['min_reward_val']))

        self.lock_temp = threading.RLock()
        self.temp = []

        self.lock_replay = threading.RLock()
        #  self.replay = ReplayBuffer(cfg, self.batch_size * task.max_n_episode() // self.n_step)
        self.replay = task.make_replay_buffer(cfg, self.batch_size * task.max_n_episode() // self.n_step)
        self.stop = False
        self.critic = model.new(task, self.cfg, "%i_%i"%(task.xid, xid))
        self.xid = xid

        self.curiosity = CuriosityPrio(task, cfg)

    def turnon(self):
        self.stop = False
        self.samplers = [
                threading.Thread(target=self.__sample_loop) for _ in range(self.cfg['sampler_scale_factor'])]
        for sampler in self.samplers:
            sampler.start()

    def shutdown(self):
        self.stop = True
        for sampler in self.samplers:
            sampler.join()

    def batch(self):
        while not self.stop:
            with self.lock_batch:
                if None != self.batches:
                    break
            time.sleep(self.cfg['exp_sampler_wait'])

        with self.lock_batch:
            if None != self.batches:
                return self.batches#[-1] # better to have something in queue always :)
        return np.vstack([[]] * 6)

    def inject(self, states, rewards, actions, features, n_states, n_rewards, n_features):
        """
        fast training ~ improvizing right after first experiences
            ~ now we dont want to traing just store those transitions
            ~ and actor will ping us back once he want to do fast training
            ~ there we evaluate our minibatch of experiences ~ fast training
            ~ if we will do training here, our gradients will be not uniform wrt our latest ciric
        """
        fast_experience = np.vstack(
            map(lambda i: (
                states[i],
                actions[i],
                features[i],
                n_states[i],
                n_rewards[i],
                n_features[i]), range(len(states))))

        with self.lock:
            if not len(self.fast_experience):
                self.fast_experience = fast_experience.T
            else:
                self.fast_experience = np.vstack([
                    self.fast_experience.T,
                    fast_experience]).T

            self.curiosity.update(states, n_states, actions)

            # will be interesting to retrain now with GAE approach
            # approximate n_reward only for last element
            # discount all on behalf of that
            # train on this :) ~ worth to try, alternative to fit toward 'real' values
            # at the end of the training with self.cfg['tau_final'] or tau_adapt ?
        #  if self.cfg['adapt_fit'] and len(states) > 1:
        #      final_reward = self.critic.predict_future(states[-1], actions[-1], features[-1])
        #      rewards = self.__discount_rewards(list(rewards[:-1]) + list(final_reward), self.discount)
        #      self.critic.fit(
        #              np.vstack(states[:-1]),
        #              np.vstack(actions[:-1]),
        #              np.vstack(features[:-1]),
        #              np.vstack(rewards[:-1]), self.cfg['tau_adapt'])

        time.sleep(self.cfg['execution_slow_down'])

    def train(self, states, rewards, actions, features, n_states, n_rewards, n_actions, n_features):
        """
        storing transitions to replay buffer
        train critic on current policy output
            ~ without this will work, however critic will be off from predictions
            ~ here maybe will become handle to update it just by TAU
                ~ pay less attention to update to this than to following :
        evaluate critic how optimal it is
            ~ important is here that we dont want to critic overfit to actor
            ~ that will lead to critic predicting actor too soon
            ~ instead, we want our critic to believe he is optimal
            ~ and update that belief based on observations
                ~ as critic will give hint to actor
                ~ if critic is optimal actor will not change critic believes
                ~ otherwise critic knows that he is not optimal and need to close gap
        """
        #  assert len(states), "training with 0 samples .. "
        if not len(states):
            return

        rewards = self.__discount_rewards(rewards, self.discount)

        with self.lock_replay:
            prios = self.curiosity.weight(states, n_states, np.vstack(actions).reshape(len(actions), 1, -1))

        with self.lock_temp:
            self.temp.append([
                states,
                rewards,
                actions,
                features,
                n_states,
                n_rewards,
                n_features,
                prios])

        if self.cfg['adapt_fit']:
            self.critic.fit( # lets bias it towards real stuff ...
                    np.vstack(states),
                    np.vstack(actions),
                    np.vstack(features),
                    np.vstack(rewards), self.cfg['tau_adapt'])

        future = self.critic.predict_future(np.vstack(n_states), np.vstack(n_actions), np.vstack(n_features))
        td_targets = n_rewards + (self.discount ** self.n_step) * future
        self.critic.fit( # lets bias it towards own stuff ...
                np.vstack(states),
                np.vstack(actions),
                np.vstack(features),
                np.vstack(td_targets), self.cfg['tau_adapt'])

    def eval(self, args):
        states, actions, action1, features, n_states, n_rewards, actionn, n_features = args

        # debug reasons
        assert len(n_features) == len(features), "features number missmatch!!"
        assert len(states) == len(n_states), "states-n_states number missmatch!!"

        if len(n_features) != len(features):
            return

        # only actions holds gradients ( action1 ) not features!
        present = self.critic.predict_present(states, action1, features) # grads are active now
        future = self.critic.predict_future(n_states, actionn, n_features)
        td_targets = n_rewards + (self.discount ** self.n_step) * future
        if self.cfg['advantages_enabled']:
            td_error = present - torch.DoubleTensor(td_targets).to(self.device)
            if self.cfg['advantages_boost']:
                for i, e in enumerate(td_error):
                    td_error[i] = e if abs(e) > 1e-2 else present[i]
                    if abs(e) < 1e-2: print("=================>", e, present[i])
        else:
            td_error = present

        self.debug_out_ex = "-> A: {:2f} T:B<{}:{}> <----".format(
                td_error[-1].item(), future[-1].item(), present[-1].item())

        self.debug_out = "-> A: {} tgt-base<{}:{}> -> R + V {} {} <----".format(
                td_error[-1], future[-1], present[-1], n_rewards[-1], td_targets[-1])

        ind = list(filter(
            lambda i: not math.isnan(td_targets[i]) and not math.isnan(td_error[i]),
            range(len(td_error))))

        # here actions represent actions which was taken, not which was output from policy ( may be the same however .. )
        self.critic.fit(states[ind], actions[ind], features[ind], td_targets[ind], self.cfg['tau_base'])
        return td_error[ind]

    def __sample_loop(self):
        while not self.stop:
            self.__push_experience()

            batch = self.__fast_exp_w_delay()
            if not len(batch):
                continue

            samples = self.__select()
            if len(samples):
                batch = np.vstack([batch, samples])#samples#

            #  print("sampling", batch.shape)

            batch = batch.T
            #  continue
            with self.lock_batch:
                self.batches = batch#np.zeros(shape = [35, 5]))

    def __fast_exp_w_delay(self):
        batch = self.__fast_exp()
        if len(batch):
            return batch
        # seems we are faster than episode execution...
        time.sleep(self.cfg['exp_sampler_wait'])
        return []

    def __fast_exp(self):
        with self.lock:
            if not len(self.fast_experience):
                return []
            if not len(self.replay) and self.fast_experience.shape[1] < self.batch_size:
                return []
            batch = np.vstack(zip(*self.fast_experience))
            self.fast_experience = []
        return batch

    def __select(self):
        if len(self.replay) < self.batch_size:
            return []

        with self.lock_replay:
            states, rewards, actions, features, n_states, n_rewards, n_features = self.replay.sample(self.batch_size)
            if not len(states):
                return []
            prios = self.curiosity.weight(states, n_states, actions)
            self.replay.update(prios)

        return np.vstack(
            map(lambda i: (
                states[i],
                actions[i],
                features[i],
                n_states[i],
                n_rewards[i],
                n_features[i]), range(len(states))))

    # well some overengineering from the past when finding perf bottleneck
    # for simplicity better to do with one lock, more readable code
    # for perf rewrite it in rust with proper locks + architecture + threading ...
    def __push_experience(self):
        with self.lock_temp:
            if not len(self.temp):
                return
            exp = self.temp.pop(-1)
        with self.lock_replay:
            self.replay.add(*exp)

    def __discount_rewards(self, r, gamma):
        """
        msft reinforcement learning explained course code, GAE approach
        """
        discounted_r = np.zeros(len(r))
        running_add = 0.
        for t in reversed(range(0, len(r))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
