import numpy as np
import time, random

import torch
from torch.multiprocessing import SimpleQueue, Process, Lock, Queue

from utils.learning_rate import LinearAutoSchedule as LinearSchedule

from utils.bot import Bot
from utils.botproxy import BotProxy

from utils import policy

from agent.simulation import simulation_launch

from threading import Thread

# multi process worker
def agent_launch(bot_id, cfg, task_factory, encoder, Actor, Critic, stop_q, callback = None, goal_encoder = None):
    agent = Zer0Bot(bot_id, cfg, task_factory, encoder, Actor, Critic, goal_encoder)

    loss_gate, mcts, signal = zip(*[(
        Queue(), SimpleQueue(), SimpleQueue()
        ) for i in range(cfg['n_simulations'])])

    sims = [ Thread(#Process(#
        target=simulation_launch,
        args=(cfg, agent.bot, bot_id, i, task_factory, loss, seed, sig, )
        ) for i, (loss, seed, sig) in enumerate(zip(loss_gate, mcts, signal)) ]

    for sim in sims:
        sim.start()

    while stop_q.empty():
        scores = agent.train(loss_gate, mcts, signal)
        if None == callback:
            continue
        scores = callback(agent, scores)
        if scores is None:
            continue
        stop_q.put(scores)

    print("AGENT OVER")
    for seed, sim in zip(mcts, sims):
        seed.put(None)
        sim.join()

    for qs in [loss_gate, mcts, signal]:
        for q in qs:
            while not q.empty():
                q.get()
    # seems queues has still something to process .. check more python multiproc to avoid this timer ...
    time.sleep(3)#10


class Zer0Bot:
    def __init__(self, bot_id, cfg, task_factory, encoder, Actor, Critic, goal_encoder):
        self.cfg = cfg
        self.bot = Bot(
                cfg,
                bot_id,
                encoder,
                goal_encoder,
                Actor,
                Critic,
                task_factory.state_size,
                task_factory.action_size,
                task_factory.wrap_action,
                task_factory.wrap_value)
        self.bot.share_memory() # !! must be done from main process !!

        self.iter = 0
        self.freezed = 0

        self.counter = 1
        self.tau = LinearSchedule(cfg['tau_replay_counter'], cfg['tau_base'], cfg['tau_final'])

        self.lock = Lock()
        self.bot = BotProxy(self.lock, cfg, self.bot, cfg['device'])

    def act(self, goal, state, history): # TODO rename to exploit
        return self.bot.exploit(goal, state, history)

    def train(self, loss, mcts, signal):
        self._encoder_freeze_schedule()
        #  seed = [random.randint(0, self.cfg['mcts_random_cap'])] * self.cfg['mcts_rounds']
        for c in mcts: # maybe we want to * n_episodes ...
            c.put([random.randint(0, self.cfg['mcts_random_cap'])] * self.cfg['mcts_rounds'])

        while all(c.empty() for c in signal):
            self._train_worker(loss)

        scores = []
        for s in signal:
            scores += s.get()
        return scores

    def _train_worker(self, loss):
        time.sleep(.1)
        status = self._update_policy(self.tau.value(), loss)
        if not status:
            return
        self.counter += 1

    def _update_policy(self, tau, loss_gate):
        if any(c.empty() for c in loss_gate):
            return False

        states, grads, actions = zip(*map(
            lambda i: self._get_grads(i, loss_gate[i].get()), range(self.cfg['n_simulations'])))

        if self.cfg["attention_enabled"]:
            # ok we will scatter additional info which is hard to be weighted w/o additional info
            gran = min(map(len, grads))
            states = np.vstack([s[:gran] for s in states])
            actions = np.vstack([a[:gran] for a in actions])
            grads = torch.cat([g[:gran] for g in grads])
        else:
            states = np.vstack(states)
            actions = np.vstack(actions)
            grads = torch.cat(grads)

        # in case of PPO it is safe to move full force
        tau = 1. if not self.cfg['ddpg'] else tau * (0 == self.counter % self.cfg['actor_update_delay'])
        self.bot.learn_actor(states, grads, actions, tau)
        return True

    def _get_grads(self, i, s_f_a_p_td):
        # this is ok to call, as we just using content which is known at creation time ( immutable )
        s, w, a = self._qa_function(i, *s_f_a_p_td)
        return s, w, a

    def _qa_function(self, objective_id, goals, states, history, actions, probs, td_targets):
        qa, dist = self.bot.q_explore(objective_id, goals, states, history)

        loss = self._qa_error(qa, td_targets)
        if self.cfg['normalize_advantages']:
            loss = policy.normalize(loss)

        probs = np.vstack(probs)
        grads = policy.policy_loss(
                torch.tensor(probs),
                dist.log_prob(torch.tensor(actions)),
                loss,
                self.cfg['ppo_eps'], self.cfg['dbgout_ratio'])

        return states, grads, actions

    def _qa_error(self, qa, td_targets):
        if not self.cfg['advantages_enabled']:
            return qa

        td_error = torch.tensor(td_targets).to(qa.device) - qa
# in case of ddpg ~ we calc advantage bit differently ~ check _eval + what is feeded here,
# turned table basically to favor of perf, basically we calculating grads w.r.t. other action
        if self.cfg['ddpg']:
            td_error = -td_error

        if not self.cfg['advantages_boost']:
            return td_error

        for i, e in enumerate(td_error):
            td_error[i] = e if abs(e) > 1e-5 else qa[i]
        return td_error

    def _encoder_freeze_schedule(self):
        if not self.cfg['freeze_delta']:
            return
        self.iter += (0 == self.freezed)
        if self.iter % self.cfg['freeze_delta']:
            return
        if not self.freezed:
            self.bot.freeze_encoders()
        self.freezed += 1
        if self.freezed <= self.cfg['freeze_count']:
            return
        self.freezed = 0
        self.bot.unfreeze_encoders()
