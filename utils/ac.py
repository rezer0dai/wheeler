import numpy as np
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, encoder, goal_encoder, actor, critic):
        super().__init__()
        self.norm = encoder # major benefit of this acrchitecture; shared encoder/normalizer!
        self.goal = goal_encoder
        self.actor = actor
        self.critic = critic

        for i, actor in enumerate(self.actor):
            self.add_module("actor_%i"%i, actor)
        for i, critic in enumerate(self.critic):
            self.add_module("critic_%i"%i, critic)

        self.norm_grads = [ p.requires_grad for p in self.norm.parameters()]
        self.goal_grads = [] if not self.goal else [ p.requires_grad for p in self.goal.parameters()]

    def share_memory(self):
        self.norm.share_memory()
        if self.goal is not None:
            self.goal.share_memory()
        for actor in self.actor:
            actor.share_memory()
        for critic in self.critic:
            critic.share_memory()

    def forward(self, a_ind, c_ind, goals, states, history_context):
        states, _ = self.norm(states, history_context)
        if self.goal is not None:
            goals = self.goal(goals, None)
 # for DQNs just update neural net to wrap actions into DQNDist wrapper
        dist = self.actor[a_ind](goals, states)
        pi = dist.sample()
        qa = self.critic[c_ind](goals, states, pi)
        return qa, dist

    def parameters(self):
        # to decide who will be responsible for norm updates!
        # select one or both ( critic_, actor_ ) to return also norm params!

        # meanwhile i prefer only actor to update norm, and critics trying to stabilize
        # without updating norm itself, and adapt to moving actors 'preference'
        assert False, "should not be accessed!" # temporary for testing

    def actor_parameters(self):
        return np.concatenate([
            list(filter(lambda p: p.requires_grad, self.norm.parameters())),
            list(filter(lambda p: p.requires_grad, self.goal.parameters())) if self.goal is not None else [],
            np.concatenate([list(actor.parameters()) for actor in self.actor])])

# originally i wanted nom + goal to optimize only at actor
# however when we do freezing of norm+goal, then my intuition now telling me enable in both is wanted
# however need to properly test trolol ...
    def critic_parameters(self):
        return np.concatenate([
            #  list(filter(lambda p: p.requires_grad, self.norm.parameters())),
            #  list(filter(lambda p: p.requires_grad, self.goal.parameters())) if self.goal is not None else [],
            np.concatenate([list(critic.parameters()) for critic in self.critic])])

    def action(self, ind, goals, states, history_context):
        states, history_context = self.norm(states, history_context)
        if self.goal is not None:
            goals = self.goal(goals, None)
        pi = self.actor[ind](goals, states)
        return pi, history_context

    def value(self, ind, goals, states, history_context, actions):
        states, _ = self.norm(states, history_context)
        if self.goal is not None:
            goals = self.goal(goals, None)
        return self.critic[ind](goals, states, actions)

    def extract_features(self, states):
        return self.norm.extract_features(states)

    def process_goals(self, goals):
        if self.goal is not None:
            goals = self.goal(goals, None)
        return goals

    def freeze_encoders(self):
        for p in self.norm.parameters():
            p.requires_grad = False
        if self.goal is None:
            return
        for p in self.goal.parameters():
            p.requires_grad = False

    def unfreeze_encoders(self):
        for g, p in zip(self.norm_grads, self.norm.parameters()):
            p.requires_grad = g
        if self.goal is None:
            return
        for g, p in zip(self.goal_grads, self.goal.parameters()):
            p.requires_grad = g
