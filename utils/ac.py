import numpy as np
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, encoder, actor, critic):
        super().__init__()
        self.norm = encoder # major benefit of this acrchitecture; shared encoder/normalizer!
        self.actor = actor
        self.critic = critic

    def share_memory(self):
        self.norm.share_memory()
        for actor in self.actor:
            actor.share_memory()
        for critic in self.critic:
            critic.share_memory()

    def forward(self, a_ind, c_ind, states, history_context):
        states, _ = self.norm(states, history_context)
 # for DQNs just update neural net to wrap actions into DQNDist wrapper
        dist = self.actor[a_ind](states)
        pi = dist.sample()
        qa = self.critic[c_ind](states, pi)
        return qa, dist

    def parameters(self):
        # to decide who will be responsible for norm updates!
        # select one or both ( critic_, actor_ ) to return also norm params!

        # meanwhile i prefer only actor to update norm, and critics trying to stabilize
        # without updating norm itself, and adapt to moving actors 'preference'
        assert False, "should not be accessed!" # temporary for testing

    def actor_parameters(self):
        return np.concatenate([
            list(self.norm.parameters()),
            np.concatenate([list(actor.parameters()) for actor in self.actor])])

    def critic_parameters(self):
        return np.concatenate([
#            list(self.norm.parameters()),
            np.concatenate([list(critic.parameters()) for critic in self.critic])])

    def action(self, ind, states, history_context):
        states, history_context = self.norm(states, history_context)
        pi = self.actor[ind](states)
        return pi, history_context

    def value(self, ind, states, history_context, actions):
        states, _ = self.norm(states, history_context)
        return self.critic[ind](states, actions)

    def features(self, states):
        return self.norm.extract_features(states)
