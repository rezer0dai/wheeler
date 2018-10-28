import math, random, time, threading
import numpy as np

import torch
from torch.autograd import Variable
import torch.multiprocessing

from collections import deque
from utils.curiosity import CuriosityPrio

from baselines.common.schedules import LinearSchedule

from utils import policy

class Critic(torch.multiprocessing.Process):
    def __init__(self, cfg, model, task, xid, stop, exps, review, comitee, grads, stats, complete, actor):
        super(Critic, self).__init__()

        self.actor = actor

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

        assert not self.cfg['gae'] or self.cfg['n_step'] == 1, "gae is currently enabled only with one step lookahead!"

        self.n_step = self.cfg['n_step']
        self.discount = self.cfg['discount_rate']
        self.n_discount = 1. if self.cfg['gae'] else (self.discount ** self.n_step)
        self.batch_size = self.cfg['batch_size']

        self.counter = 0
        self.tau = LinearSchedule(cfg['tau_replay_counter'],
               initial_p=self.cfg['tau_base'],
               final_p=cfg['tau_final'])

        self.lock = threading.RLock() # we protect only write operations ~ yeah yeah not so good :)
        self.replay = task.make_replay_buffer(cfg)

        self.model = model.new(task, self.cfg, "%i_%i"%(task.xid, xid))
        self.model.share_memory()

        self.curiosity = CuriosityPrio(task, cfg)

    def q_a_function(self, states, actions, features, td_targets):
# even in multiprocess critic is shared from main process
        q_a = self.model.predict_present(states, actions, features) # grads are active now
        if not self.cfg['advantages_enabled']:
            return q_a

        td_error = torch.tensor(td_targets).to(self.device) - q_a
        # in case of ddpg ~ we calc advantage bit differently ~ check _eval + what is feeded here,
        # turned table basically to favor of perf, *should* (hehe) be equivalent with original
        if actions.requires_grad:
            td_error = -td_error

        if not self.cfg['advantages_boost']:
            return td_error

        for i, e in enumerate(td_error):
            td_error[i] = e if abs(e) > 1e-3 else q_a[i]
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

            full, info, exp = exp
            if not full:
                self._inject(info, exp) # action
            else:
                self._train(info, exp) # delta

            if not self.cfg['critic_learn_delta']:
                continue
            if not len(self.fast_experience):
                continue
            if self.fast_experience.shape[1] < self.cfg['critic_learn_delta']:
                continue

            for batch in self._do_sampling():
                self.review.put(batch)

    def _inject(self, action, exp):
#        return
        states, rewards, actions, probs, features, n_states, n_features = exp
        actions, _, _, s_norm, n_norm = self.stack_inputs(actions, states, n_states)

        n_rewards = policy.td_lambda(rewards, self.n_step, self.discount) if not self.cfg['gae'] else policy.gae(
                rewards, 
                # here we need to append after state
                self.model.predict_future(
                    np.vstack([s_norm, [n_norm[-1-i] for i in range(self.n_step)]]), 
                    np.vstack([actions, np.vstack([action] * self.n_step)]), 
                    np.vstack(list(features) + [n_features[-1-i] for i in range(self.n_step)])),
                self.discount, self.cfg['gae_tau'])

        delta = len(states) % self.cfg['n_critics']
        fast_experience = np.vstack(
            map(lambda i: (
                s_norm[i],
                actions[i],
                probs[i],
                features[i],
                n_norm[i],
                n_rewards[i],
                n_features[i]), range(
                    ((delta % self.cfg['n_critics']) + self.xid - 1) % self.cfg['n_critics'],
                    len(states), 
                    self.cfg['n_critics'] if self.cfg['disjoint_critics'] else 1)))

        if not len(self.fast_experience):
            self.fast_experience = fast_experience.T
        else:
            self.fast_experience = np.vstack([
                self.fast_experience.T,
                fast_experience]).T

    def _train(self, delta, exp): 
        states, rewards, actions, probs, features, n_states, n_features = exp
        if not len(states):
            return

        actions, states, n_states, s_norm, n_norm = self.stack_inputs(actions, states, n_states)

        features = np.vstack(features)
        n_rewards = policy.td_lambda(rewards, self.n_step, self.discount) if not self.cfg['gae'] else policy.gae(
                rewards, self.model.predict_future(s_norm, actions, features), self.discount, self.cfg['gae_tau'])

        self._update_memory(states, rewards, actions, probs, features,
                n_states, n_rewards, n_features,
                s_norm, n_norm, delta)

        self._update_normalizer(states)

        self._td_lambda()

        if not self.cfg['reinforce_clip']:
            return

# else we will do reinforce # i did not tried this part properly yet ..
        rewards = np.asarray(
                policy.discount(rewards, self.discount))

        self.model.fit( # lets bias it towards real stuff ...
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
                    random.sample(range(len(states)), random.randint(1, len(states) - 1))])
        if len(self.replay) < self.batch_size:
            return None
        s, _, _, _, _, n, _, _ = self.replay.sample(self.batch_size, None)
        self.task.update_normalizer(s)
        self.task.update_normalizer(n)

    def _update_memory(self,
            states, rewards, actions, probs, features,
            n_states, n_rewards, n_features,
            s_norm, n_norm, delta):

        prios = self.curiosity.weight(s_norm, n_norm, actions)
        with self.lock:
            self.replay.add(
                map(lambda i: (
                    states[i],
                    rewards[i],
                    actions[i],
                    probs[i],
                    features[i],
                    n_states[i],
                    n_rewards[i],
                    n_features[i]), range(len(states))),
                prios, delta)
        self.curiosity.update(s_norm, n_norm, actions)

    def _eval(self, args):
        states, actions, features, n_states, n_action, n_features, n_rewards = args

        assert len(n_features) == len(features), "features missmatch"
        if len(n_features) != len(features):
            return

        future = self.model.predict_future(n_states, n_action, n_features)
        td_targets = n_rewards + self.n_discount * future

        self.counter += 1
        self.model.fit(states, actions, features, td_targets,
                self.tau.value(self.counter) * (0 == self.counter % self.cfg['critic_update_delay']))

        self.debug_out_ex = "[ TARGET:{:2f} replay::{} ]<----".format(
                td_targets[-1].item(), len(self.replay))

        return td_targets

    def _do_sampling(self):
        batch = self._fast_exp()
        if None == batch:
            return

#        first_order_experience_focus = '''
        for _ in range(self.cfg['fast_exp_epochs']):
            samples = self._select()
            mini_batch = batch if None == samples else np.vstack([batch, samples])
            population = random.sample(
                    range(len(mini_batch)),
                    random.randint(
                        1 + (len(mini_batch) - 1) // (1 + self.cfg['fast_exp_epochs']),
                        len(mini_batch) - 1))
            yield mini_batch[population].T

        replay_focused = '''
        for _ in range(self.cfg['fast_exp_epochs']):
            population = random.sample(
                    range(len(batch)),
                    random.randint(
                        1 + (len(batch) - 1) // (1 + self.cfg['fast_exp_epochs']),
                        len(batch) - 1))
            samples = self._select()
            if None != samples:
                yield np.vstack([batch[population], samples]).T
            else:
                yield batch[population].T
#        '''

        yield batch.T

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

        with self.lock:
            return self._locked_select()

    def _locked_select(self):
        states, _, actions, probs, features, n_states, n_rewards, n_features = self.replay.sample(
                self.batch_size, self if not self.cfg['disjoint_critics'] else None) # we can not re-evaluate on incomplete episode data ...
        if not len(actions):
            return None

# ah uh, we would like to skip stacking, unnecessary perf overhead...
        actions, _, _, s_norm, n_norm = self.stack_inputs(actions, states, n_states)

        prios = self.curiosity.weight(s_norm, n_norm, actions)
        if self.cfg['replay_cleaning']:
 # seems we are bit too far for PG to do something good, replay buffer should abandond those
            prios[self.cfg['prob_treshold'] < np.abs(probs)] = 0

        self.replay.update(prios)

        return np.vstack(
            map(lambda i: (
                s_norm[i],
                actions[i],
                probs[i],
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

# main bottleneck of whole solution, but we experimenting so .. :)
# also i think can be played with, when enough hardware/resources 
#  -> properly scale, and do thinks on background in paralel..
# + if main concern is speed i would not do it in python in first place ..
    def reanalyze_experience(self, episode, indices):
        states = np.asarray([e[0][0] for e in episode])
        s_norm = self.task.normalize_state(states)

        actions = [e[0][2] for e in episode]
        f, p = self.actor.reevaluate(s_norm, actions)

        rewards = np.asarray(self.task.update_goal(
            *zip(*[( # magic *
                e[0][1], # rewards .. just so he can forward it to us back
                e[0][0], # states .. 
#                e[0][2], # action .. well for now no need, however some emulator may need them
                bool(random.randint(0, self.cfg['her_max_ratio'])), # update or not
                ) for i, e in enumerate(episode)])))

        n = policy.td_lambda(rewards, self.n_step, self.discount) if not self.cfg['gae'] else policy.gae(
                rewards, 
                self.model.predict_future(s_norm, np.asarray(actions), np.asarray(f)), 
                self.discount, self.cfg['gae_tau'])

        return [(
            e[0][0], # states we dont change them
            e[0][1], # rewards -> imho no need to return in current implementation..
            e[0][2], # actions we dont change them
            p[indices[i]], 
            f[indices[i]], 
            e[0][5], # n-states we dont change them
            n[indices[i]],
            f[(indices[i] + self.n_step) if indices[i]+self.n_step < len(f) else -1]
            ) for i, e in enumerate(map(lambda j: episode[j], indices))]
