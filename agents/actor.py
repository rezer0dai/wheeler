import numpy as np

class Actor:
    def __init__(self, model):
        self.actor = model

    def share_memory(self):
        self.actor.share_memory()

    def reload(self, master):
        self.actor.set(master.load())
    def load(self):
        return self.actor.get()

    def learn(self, states, advantages, actions, tau):
        assert len(actions) == len(states), "{} - {}".format(actions.shape, states.shape)
#        if tau: print("\n WE DO LEARN ")
        self.actor.fit(states, advantages, actions, tau)

    def get_action_w_grad(self, state, history):
        return self.actor.predict_present(state, history)

    def get_action_wo_grad(self, state, history):
        a, f = self.actor.predict_present(state, history)
        return a.detach().cpu().numpy(), f.detach().cpu().numpy()

    def predict(self, state, history):
        return self.actor.predict_future(state, history)

    def reset(self):
        self.actor.reset()

    def extract_features(self, episode, n_step, indicies):
        if not self.actor.target.recomputable():
            return map(lambda i: episode[i][0], indicies)
        states = np.asarray([e[0][0] for e in episode])
        f = self.actor.target.extract_features(states)
        return [(
            e[0][0], e[0][1], e[0][2], f[indicies[i]], e[0][4], e[0][5], f[
                indicies[i]+n_step if indicies[i]+n_step < len(f) else 0]
            ) for i, e in enumerate(map(lambda j: episode[j], indicies))]
