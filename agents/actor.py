import numpy as np
import torch

from utils import policy

class Actor:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def share_memory(self):
        self.model.share_memory()

    def reload(self, master):
        self.model.set(master.load())
    def load(self):
        return self.model.get()

    def learn(self, states, advantages, actions, tau):
        assert len(actions) == len(states), "{} - {}".format(actions.shape, states.shape)
#        if tau: print("\n WE DO LEARN ")
        self.model.fit(states, advantages, actions, tau)

    def get_action_w_grad(self, state, history):
        return self.model.predict_present(state, history)

    def get_action_wo_grad(self, state, history):
        d, f = self.model.predict_present(state, history)
        return d, f.detach().cpu().numpy()

    def predict(self, state, history):
        dist, features = self.model.predict_future(state, history)
        return dist.sample().detach().cpu().numpy(), features.detach().cpu().numpy()

    def reevaluate(self, states, actions):
        f = self.model.explorer.extract_features(states)
        p = self.model.explorer(
                torch.from_numpy(states),
                torch.tensor(f)).log_prob(torch.tensor(actions)).detach().cpu().numpy()
        return f, p
