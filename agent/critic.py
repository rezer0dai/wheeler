import random, threading, time
import numpy as np

from utils import policy
from utils.curiosity import CuriosityPrio
from utils.learning_rate import LinearAutoSchedule as LinearSchedule

def critic_launch(cfg, bot, objective_id, task_factory, update_goal, ping, sync, loss_gate, share_gate, stats):
    critic = Critic(cfg, bot, objective_id, task_factory, update_goal)
    critic.training_loop(ping, sync, share_gate, loss_gate, stats)
    print("CRITIC OVER")

class Critic:
    def __init__(self, cfg, bot, objective_id, task_factory, update_goal):
        assert not cfg['gae'] or cfg['n_step'] == 1, "gae is currently enabled only with one step lookahead!"

        self.cfg = cfg
        self.objective_id = objective_id

        self.bot = bot
        self.update_goal = update_goal

        self.stop = False

        self.debug_out_ex = "y" * 10

        self.n_step = self.cfg['n_step']
        self.discount = self.cfg['discount_rate']
        self.n_discount = 1. if self.cfg['gae'] else (self.discount ** self.n_step)
        self.batch_size = self.cfg['batch_size']

        self.counter = 0
        self.tau = LinearSchedule(cfg['tau_replay_counter'],
               initial_p=self.cfg['tau_base'],
               final_p=cfg['tau_final'])

        self.replay = task_factory.make_replay_buffer(cfg)

        self.full_episode = []
        self.last_train_cap = self.cfg['critic_learn_delta']

        # here imho configurable choise : use curiosity, td errors, random, or another method
        self.curiosity = CuriosityPrio(
                task_factory.state_size, task_factory.action_size,
                task_factory.action_range, task_factory.wrap_action, cfg['device'], cfg)

    def training_loop(self, ping, sync, share_gate, loss_gate, stats):
        while True:
            exp = share_gate.get()
            if None == exp:
                break

            full, action, exp = exp
            if not full:
                self._inject(action, exp)
            else:
                self._train(loss_gate, ping, stats, action, exp)

            if not self.cfg['critic_learn_delta']:
                continue
            if len(self.full_episode) < self.last_train_cap:
                continue

            self.last_train_cap += self.cfg['critic_learn_delta']

#            print("\n%s\nDO FAST TRAIN : %i\n%s\n"%('*' * 60, len(self.full_episode), '*' * 60))
            ping.put(True) # old style scoping ... python nicer way out there ?
            for batch in self._do_sampling():
                self._eval(loss_gate, stats, batch)
            ping.get()

        self._dtor(ping, sync, stats)

    def _dtor(self, ping, sync, stats):
        self.stop = True
        while not ping.empty():
            time.sleep(.1)
        while not stats.empty():
            stats.get()
        sync.put(True)

    def _inject(self, action, exp):
        goals, states, features, actions, probs, rewards, n_goals, n_states, n_features, good = exp
        if not len(states):
            return

        n_rewards = policy.td_lambda(rewards, self.n_step, self.discount) if not self.cfg['gae'] else policy.gae(
                rewards, self.bot.qa_future(
                    self.objective_id,
                    np.vstack([goals, [goals[-1]]]).reshape(len(goals) + 1, -1),
                    np.vstack([states, n_states[-1]]),
                    np.vstack([features, [n_features[-1]]]),
                    np.vstack([actions, [action]])),
                self.discount, self.cfg['gae_tau'], stochastic=False)

        full_episode = np.vstack(zip(*[goals, states, features, actions, probs, rewards, n_goals, n_states, n_features, n_rewards, good]))

        if not len(self.full_episode):
            self.full_episode = full_episode
        else:
            self.full_episode = np.vstack([self.full_episode, full_episode])

    def _train(self, loss_gate, ping, stats, action, exp):
        self._inject(action, exp)

        self._update_memory()

        self._self_play(loss_gate, ping, stats)
        # abandoned reinforce clip, as i think that is no go for AGI...

#        print("\n%s\nFULL EPISODE LENGTH : %i\n%s\n"%('*' * 60, len(self.full_episode), '*' * 60))
        self.full_episode = []
        self.last_train_cap = self.cfg['critic_learn_delta']

    def _self_play(self, loss_gate, ping, stats):
        ping.put(True)
        for _ in range(self.cfg['full_replay_count']):
            samples = self._select()
            if None == samples:
                continue
            self._eval(loss_gate, stats, samples.T)
        ping.get()

    def _update_memory(self):
        goals, states, features, actions, probs, rewards, n_goals, n_states, n_features, n_rewards, good = self.full_episode.T
        goals, states, n_goals, n_states, actions = np.vstack(goals), np.vstack(states), np.vstack(n_goals), np.vstack(n_states), np.vstack(actions)

        prios = self.curiosity.weight(states, n_states, actions)

        self.replay.add(
            map(lambda i: (
                goals[i], states[i], features[i], actions[i], probs[i], rewards[i],
                n_goals[i], n_states[i], n_features[i], n_rewards[i]
                ), filter(
                    lambda i: bool(sum(good[i:i+self.cfg['good_reach']])),
                    range(len(states)))),
            prios, hash(states.tostring()))

        self.curiosity.update(states, n_states, actions)

    def _eval(self, loss_gate, stats, args):
        if self.stop:
            return

        goals, states, features, actions, probs, n_goals, n_states, n_features, n_rewards = args

        assert len(n_features) == len(features), "features missmatch"
        if len(n_features) != len(features):
            return

        goals, states, features, actions = np.vstack(goals), np.vstack(states), np.vstack(features), np.vstack(actions)
        n_goals, n_states, n_features, n_rewards = np.vstack(n_goals), np.vstack(n_states), np.vstack(n_features), np.vstack(n_rewards)

        # func approximators; self play
        n_qa = self.bot.q_future(self.objective_id, n_goals, n_states, n_features)
        # n_step target + bellman equation
        td_targets = n_rewards + self.n_discount * n_qa
        # learn !!
        self.counter += 1
        self.bot.learn_critic(self.objective_id, goals, states, features, actions, td_targets,
                self.tau.value() * (0 == self.counter % self.cfg['critic_update_delay']))

# propagate back to simulation ~ debug purposes
        if None != stats and self.cfg['dbgout'] and not self.stop:
            stats.put("[ TARGET:{:2f} replay::{} ]<----".format(
                    td_targets[-1].item(), len(self.replay)))

# propagate back to main process
        loss_gate.put([ goals, states, features, actions, probs, td_targets ])

        # WARNING : EXPERIMENT ~~> here we on purpose provide same features as for n-state
            # basically we are leaking future of that trajectory, what our agent will do ?
            # bellman will be probably not proud of me at this point :)
        #loss_gate.put([ goals, states, n_features, actions, probs, td_targets ])

    def _population(self, batch):
        return random.sample(range(len(batch)), random.randint(
            1, min(2 * self.cfg['batch_size'], len(batch) - 1)))

    def _do_sampling(self):
        if self.stop:
            return
        batch = self._fast_exp()
        if None == batch:
            return

#        first_order_experience_focus = '''
        for _ in range(self.cfg['fast_exp_epochs']):
            samples = self._select()
            mini_batch = batch if None == samples else np.vstack([batch, samples])
            population = self._population(mini_batch)
            yield mini_batch[population].T

        replay_focused = '''
        for _ in range(self.cfg['fast_exp_epochs']):
            population = self._population(batch)
            samples = self._select()
            if None != samples:
                yield np.vstack([batch[population], samples]).T
            else:
                yield batch[population].T
#        '''
        population = self._population(batch)
        yield batch[population].T # push towards latest experience

    def _fast_exp(self):
        if max(len(self.replay), len(self.full_episode)) < self.batch_size:
            return None

        goals, states, features, actions, probs, _, n_goals, n_states, n_features, n_rewards, _ = self.full_episode.T
        return np.vstack(zip(goals, states, features, actions, probs, n_goals, n_states, n_features, n_rewards))

    def _select(self):
        if len(self.replay) < self.batch_size:
            return None

        data = self.replay.sample(self.batch_size, self)
        if None == data:
            return None

        goals, states, features, actions, probs, _, n_goals, n_states, n_features, n_rewards = data
        if not len(actions):
            return None

        self._update_replay_prios(states, n_states, actions)

        return np.vstack(zip(goals, states, features, actions, probs, n_goals, n_states, n_features, n_rewards))

    def _update_replay_prios(self, states, n_states, actions):
        if not self.cfg['replay_cleaning']:
            return
        states, n_states, actions = np.vstack(states), np.vstack(n_states), np.vstack(actions)
        prios = self.curiosity.weight(states, n_states, actions)
# seems we are bit too far for PG ( PPO ) to do something good, replay buffer should abandon those
        prios[self.cfg['prob_treshold'] < np.abs(np.vstack(probs).mean(-1))] = 0
        self.replay.update(prios)

# main bottleneck of whole solution, but we experimenting so .. :)
# also i think can be played with, when enough hardware/resources
#  -> properly scale, and do thinks on background in paralell..
# + if main concern is speed i would not do it in python in first place ..
    def reanalyze_experience(self, episode, indices, recalc):
        # imho i iterate too much trough episode ... better to implement it in one sweep ... TODO
        goals, states, f, a, p = zip(*[
                [e[0][0], e[0][1], e[0][2], e[0][3], e[0][4]] for e in episode ])

        goals, states = np.asarray(goals), np.asarray(states)

        if recalc:
            f, p = self.bot.reevaluate(self.objective_id, goals, states, a)

        r, g, s, n_g, n_s = zip(*self.update_goal(
            *zip(*[( # magic *
                e[0][5], # rewards .. just so he can forward it to us back
                e[0][0], # goals ..
                e[0][1], # states ..
                e[0][6], # n_goals ..
                e[0][7], # n_states ..
#                e[0][2], # action .. well for now no need, however some emulator may need them
                bool(random.randint(0, self.cfg['her_max_ratio'])), # update or not
                ) for e in episode])))

        n = [ e[0][9] for e in episode ] if not recalc or not self.cfg['gae'] else policy.gae(
                r,
                self.bot.qa_future(self.objective_id, goals, states, np.asarray(f), np.asarray(a)),
                self.discount, self.cfg['gae_tau'])

        for i in indices:
            yield ( g[i], s[i], f[i], a[i], p[i], r[i],
                    n_g[i], n_s[i], f[(i + self.n_step) if i+self.n_step < len(f) else -1], n[i] )
