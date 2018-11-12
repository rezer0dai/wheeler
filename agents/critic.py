import math, random, time, threading
import numpy as np

import torch
from torch.autograd import Variable
import torch.multiprocessing

from collections import deque
from utils.curiosity import CuriosityPrio

from utils.learning_rate import LinearAutoSchedule as LinearSchedule

from utils import policy

class Critic(torch.multiprocessing.Process):
    def __init__(self,
            cfg, model, task, task_info,
            bot_id, objective_id, xid,
            stop, exps, review, comitee, grads, stats, complete,
            actor):

        super().__init__()

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
        self.objective_id = objective_id

        self.task = task
        self.full_episode = []
        self.last_train_cap = self.cfg['critic_learn_delta']

        self.device = task.device()

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
        self.replay = task_info.make_replay_buffer(self.cfg, task.objective_id)

        self.model = model

        # here imho configurable choise : use curiosity, td errors, random, or another method
        self.curiosity = CuriosityPrio(
                task_info.state_size, task_info.action_size,
                task_info.action_range, task_info.wrap_action, self.device, cfg)

    def q_a_function(self, states, actions, features, td_targets):
# even in multiprocess critic is shared from main process
        q_a = self.model.predict_present(self.xid, states, actions, features) # grads are active now
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

            full, delta, action, exp = exp
            if not full:
                self._inject(delta, action, exp) # action
            else:
                self._train(delta, action, exp) # delta

            if not self.cfg['critic_learn_delta']:
                continue
            if len(self.full_episode) < self.last_train_cap:
                continue # make sure this working lol ...

            self.last_train_cap += self.cfg['critic_learn_delta']

            print("\n%s\nDO FAST TRAIN : %i\n%s\n"%('*' * 60, len(self.full_episode), '*' * 60))
            for batch in self._do_sampling(delta):
                self.review.put(batch)

    def _inject(self, e, action, exp):
        states, rewards, actions, probs, features, n_states, n_features, good = exp
        if not len(states):
            return

        features = np.vstack(features)
        actions, _, _, s_norm, n_norm = self.stack_inputs(actions, states, n_states)
        n_rewards = policy.td_lambda(rewards, self.n_step, self.discount) if not self.cfg['gae'] else policy.gae(
                rewards, self.model.predict_future(
                    np.vstack([s_norm, n_norm[-1]]),
                    np.vstack([actions, np.vstack([action])]),
                    np.vstack([features, np.vstack([n_features[-1]])])),
                self.discount, self.cfg['gae_tau'], stochastic=False)

        full_episode = np.vstack(zip(*[states, features, actions, probs, rewards, n_states, n_features, n_rewards, good]))

        if not len(self.full_episode):
            self.full_episode = full_episode
        else:
            self.full_episode = np.vstack([self.full_episode, full_episode])

    def _train(self, delta, action, exp):
        self._inject(None, action, exp)

        self._update_memory(delta)

        self._self_play()
        # abandoned reinforce clip, as i think that is no go for AGI...

        print("\n%s\nFULL EPISODE LENGTH : %i\n%s\n"%('*' * 60, len(self.full_episode), '*' * 60))
        self.full_episode = []
        self.last_train_cap = self.cfg['critic_learn_delta']

    def _self_play(self):
        for _ in range(self.cfg['full_replay_count']):
            samples = self._select()
            if None == samples:
                continue
            self.review.put(samples.T)
        self.complete.put(True)

    def _update_memory(self, delta):
        states, features, actions, probs, rewards, n_states, n_features, n_rewards, good = self.full_episode.T
        actions, _, _, s_norm, n_norm = self.stack_inputs(actions, states, n_states)

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
                    n_features[i]), filter(lambda i: bool(sum(good[i:i+self.cfg['good_reach']])), range(len(states)))),
                prios, delta, hash(states.tostring()))
        self.curiosity.update(s_norm, n_norm, actions)

    def _eval(self, args):
        states, actions, features, n_states, n_action, n_features, n_rewards = args

        assert len(n_features) == len(features), "features missmatch"
        if len(n_features) != len(features):
            return

        future = self.model.predict_future(n_states, n_action, n_features)
        td_targets = n_rewards + self.n_discount * future

        self.counter += 1
        self.model.fit(self.xid, states, actions, features, td_targets,
                self.tau.value() * (0 == self.counter % self.cfg['critic_update_delay']))

        self.debug_out_ex = "[ TARGET:{:2f} replay::{} ]<----".format(
                td_targets[-1].item(), len(self.replay))

        return td_targets

    def _do_sampling(self, delta):
        batch = self._fast_exp(delta)
        if None == batch:
            return

#        first_order_experience_focus = '''
        for _ in range(self.cfg['fast_exp_epochs']):
            samples = self._select()
            mini_batch = batch if None == samples else np.vstack([batch, samples])
            population = random.sample(
                    range(len(mini_batch)),
                    random.randint(
                        1 + (len(mini_batch) - 1) // (1 + self.cfg['fast_exp_epochs'] // 2),
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

    def _fast_exp(self, delta):
        if max(len(self.replay), len(self.full_episode)) < self.batch_size:
            return None

        indices = range(
                    ((delta % self.cfg['n_critics']) + self.xid - 1) % self.cfg['n_critics'],
                    len(self.full_episode.T),
                    self.cfg['n_critics'] if self.cfg['disjoint_critics'] else 1)
        if len(indices) < 1:
            indices = range(len(self.full_episode.T))

        states, features, actions, probs, _, n_states, n_features, n_rewards, _ = self.full_episode[indices].T
        _, _, _, s_norm, n_norm = self.stack_inputs(actions, states, n_states)
        return np.vstack(zip(*[s_norm, actions, probs, features, n_norm, n_rewards, n_features]))

    def _select(self):
        if len(self.replay) < self.batch_size:
            return None

        with self.lock:
            return self._locked_select()

    def _locked_select(self):
        data = self.replay.sample(self.batch_size, self)
        if None == data:
            return None

        states, _, actions, probs, features, n_states, n_rewards, n_features = data
        if not len(actions):
            return None

# ah uh, we would like to skip stacking, unnecessary perf overhead...
        actions, _, _, s_norm, n_norm = self.stack_inputs(actions, states, n_states)

        prios = self.curiosity.weight(s_norm, n_norm, actions)
        if self.cfg['replay_cleaning']:
 # seems we are bit too far for PG to do something good, replay buffer should abandond those
            prios[self.cfg['prob_treshold'] < np.abs(np.vstack(probs).mean(-1))] = 0

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
        f, p = self.actor.reevaluate(self.objective_id, s_norm, actions)

        rewards, s, n_s = self.task.update_goal(
            *zip(*[( # magic *
                e[0][1], # rewards .. just so he can forward it to us back
                e[0][0], # states ..
                e[0][5], # states ..
#                e[0][2], # action .. well for now no need, however some emulator may need them
                bool(random.randint(0, self.cfg['her_max_ratio'])), # update or not
                ) for e in episode]))

        n = policy.td_lambda(rewards, self.n_step, self.discount) if not self.cfg['gae'] else policy.gae(
                rewards,
                self.model.predict_future(s_norm, np.asarray(actions), np.asarray(f)),
                self.discount, self.cfg['gae_tau'])

        return [(
            s[indices[i]],#e[0][0], # states we dont change them but update goal can change them ...
            rewards[indices[i]],#e[0][1], # rewards -> imho no need to return in current implementation..
            e[0][2], # actions we dont change them
            p[indices[i]],
            f[indices[i]],
            n_s[indices[i]],#e[0][5], # n-states we dont change them but ...
            n[indices[i]],
            f[(indices[i] + self.n_step) if indices[i]+self.n_step < len(f) else -1]
            ) for i, e in enumerate(map(lambda j: episode[j], indices))]
