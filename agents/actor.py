import numpy as np

class Actor:
    def __init__(self, model):
        self.actor = model

    def learn(self, states, advantages, actions, tau):
        assert len(actions) == len(states), "{} - {}".format(actions.shape, states.shape)
        self.actor.fit(states, advantages, actions, tau)

    def get_action_w_grad(self, state, history):
        return self.actor.predict_present(state, history)

    def get_action_wo_grad(self, state, history):
        a, f = self.actor.predict_present(state, history)
        return a.detach().cpu().numpy(), f

    def predict(self, state, history):
        return self.actor.predict_future(state, history)

    def reset(self):
        self.actor.reset()
