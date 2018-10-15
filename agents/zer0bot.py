import numpy as np
import threading, math, time, sys
from concurrent.futures import ThreadPoolExecutor, wait

from agents.actor import Actor
from agents.simulation import Simulation

import torch
from torch.autograd import Variable

from baselines.common.schedules import LinearSchedule

def safe_stack(data):
    used = False
    stack = None
    for d in data:
        if not len(d):
            continue
        if not used:
            stack = d
            used = True
        else:
            stack = np.vstack([stack, d])
    return stack

def safe_tstack(data):
    used = False
    stack = None
    for d in data:
        if not len(d):
            continue
        if not used:
            stack = d
            used = True
        else:
            stack = torch.cat([stack, d])
    return stack

def align(data):
    try:
        return safe_stack(data)#np.vstack(data)
    except:
        return safe_tstack(data)
#   return np.vstack([np.hstack(d) for d in data])
    return stafe_stack([np.hstack(d) for d in data if len(d)])

class Zer0Bot:
    # n_step -> bias vs variance hyperparam => how much to the future we estimate for advantages
    # learning_step -> how many episodes per one update of policy
    # discount -> discount factor of future rewards, well should varies of max lenght of episode
    # max_n_episode -> in non episodic task we may want to stop/pause somehwere to start learning
    def __init__(self, cfg, task, model_actor, model_critic):
        self.cfg = cfg

        if not self.cfg['threading']:
            self.cfg['learn_delta'] = 0

        self.task = [task.new(i) for i in range(task.subtasks_count())]
        self.task_main = task
        self.n_step = self.cfg['n_step']

        self.lock = threading.RLock()
        self.counter = 0
        self.tau = LinearSchedule(cfg['tau_replay_counter'],
               initial_p=self.cfg['tau_base'],
               final_p=cfg['tau_final'])


        self.model_actor = model_actor
        self.__setup_actor(model_actor)
        self.__setup_critics(model_critic)

        self.fast_learner = [threading.Thread(target=self.__learn_fast) for _ in range(self.cfg['learner_scale_factor'])]

    def act(self, state, history):
        a, history = self.actor.get_action_wo_grad(state.reshape(1, -1), history)
        return (a, history)

    def enable_fast_training(self):
#        self.fast_learner.start()
        for fl in self.fast_learner:
            fl.start()

    def train(self, n_episodes):
        self.actor.reset()
        self.__train_critics(n_episodes)
        self.__full_train()

    def __setup_actor(self, model):
        self.actor = Actor(model.new(self.task_main, self.cfg))

    def __setup_critics(self, model):
        self.simulations = [Simulation(
            self.cfg, model, self.task[i],
            self.actor.get_action_w_grad, self.actor.get_action_wo_grad, self.actor.predict,
            ) for i in range(self.task_main.subtasks_count())]

    def __learn_fast(self):
        last = 0
        while not self.task_main.learned():
            time.sleep(self.cfg['learn_loop_to'])
            if self.task_main.episode_counter() < last + self.cfg['learn_delta']:
                continue
            self.__full_train()
            last = self.task_main.episode_counter()

    def __full_train(self):
        self.counter += 1
        self.__update_policy(self.tau.value(self.counter))

    def __train_critics(self, n_episodes):
        workers = []
        pool = ThreadPoolExecutor(len(self.simulations))
        for i, simulation in enumerate(self.simulations):
#           workers.append(pool.submit(Simulation.turnon, simulation,
#               self.task[i].new(i), n_episodes, self.actor))
            simulation.turnon(self.task[i].new(i), n_episodes, self.actor)
        wait(workers)

    def __update_policy(self, tau):
        states, advantages, actions = zip(*map(
            lambda simulation: simulation.get_grads(), self.simulations))

        # following sections is really questionable, if attention model is worth
        # as otherwise we just vstack everything together, and push to nn...
        # problematic mostly at FAST learn, also it can happen some guys are too picky
        # to learn ~ good is seldomly true ..

        if self.cfg["attention_enabled"]:
            assert len(states), "nothing to learn after run.. well try again ..."
            if any(not len(s) for s in states):
                return

            if len(advantages) != len(self.simulations):
                return
            # ok we will scatter additional info which is hard to be weighted
            gran = min([len(a) for a in advantages])
            states = np.vstack([s[:gran] for s in states])
            actions = np.vstack([a[:gran] for a in actions])
            advantages = torch.stack([v[:gran] for v in advantages])
        else:
            if all(not len(a) for a in actions):
                return
            states = [s for s in states if len(states)]
            states = safe_stack(states)
            actions = align(actions)
            advantages = align(advantages)

        advantages = self.__normalize(advantages)
        self.actor.learn(states, advantages, actions, tau)

    def __normalize(self, advantages):
        """
        work over standard mean, to avoid unecessary chaos in policy, source from OpenAI
        """
        if not self.cfg['normalize_advantages']:
            return advantages
        normalize = lambda a: (a - a.mean()) / a.std()
        return normalize(advantages)
