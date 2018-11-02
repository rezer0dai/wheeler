import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class DDPGDist():
    def __init__(self, actions):
        self.actions = actions
    def log_prob(self, actions):
        return torch.ones(actions.shape)
    def sample(self):
        return self.actions.clone()

class DDPG(nn.Module):
    def __init__(self, wrap_action):
        super().__init__()
        self.wrap_action = wrap_action
    def forward(self, actions):
        actions = self.wrap_action(actions)
        dist = DDPGDist(actions)
        return dist

class PPO(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(1, action_size))
    def forward(self, mu):
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist

def normalize(loss):
    """
    work over standard mean, to avoid unecessary chaos in policy, source from OpenAI
    .. avoid using pgd_loss.mean() and use pgd_loss.sum() instead
        + may stabilize learning
        - well, it normalize our advantage ( value not anymore holds advantage, +/- can be swapped )
    .. i prefer to tune { learning rates / batch size / step per learning } instead
    """
    normalize = lambda a: (a - a.mean()) / a.std()
    return normalize(loss)

def policy_debug_out(diff):
    ratio = diff.detach().exp().mean()
    if ratio > 1.5 or ratio < .5:
        try:
            print("\n ratio --> ", ratio)
            print("\n\t results ===> ", (dist.log_prob(torch.tensor(actions)) - probs).exp()[:4])
            print("\n\t new probs ==> ", dist.log_prob(torch.tensor(actions))[:4])
            print("\n\t old probs ==> ", probs[:4])
        except:
            print("!"*40, "\n ration too low \n", "!"*40)

def ppo(diff, loss, eps):
    """ paper : https://arxiv.org/abs/1707.06347
        + using grads from policy probabilities, clamping them... 
        - however not efficient to use with replay buffer ( past action obsolete, ratio always clipped )
    """
#    print("PPO")

    ratio = diff.exp()

    surr1 = torch.clamp(ratio, min=1.-eps, max=1.+eps) * loss
    surr2 = ratio * loss
    grads = torch.min(surr1, surr2)

    return grads

def vanila_pg(probs, loss):
    """ paper : ...
        + using grads from policy probabilities, clamping them... 
        - however it can be way to obselete if replay buffer used ( big number {pos/neg} for prob ~ big change )
    """
#    print("VG")
    grads = probs * loss * .1
    return grads 

def ddpg(loss):
    """ paper : https://arxiv.org/abs/1509.02971
        + effective to use with replay buffer
        - using grads from critic not from actual pollicy ( harder to conv )
    """
    grads = loss
    return loss

def policy_loss(old_probs, new_probs, loss, ppo_eps, dbgout = True):
    if not new_probs.requires_grad: # dppg
        return ddpg(loss)

    diff = (new_probs - old_probs)

    if dbgout:
        policy_debug_out(diff)

    mean = old_probs.mean()
    if abs(mean) < 2. and abs(mean) > 1e-3:# online policy, we can use PPO
#    ratio = diff.detach().exp().mean()
#    if ration < 2. or ratio > 1e-2:
        return ppo(diff, loss, ppo_eps)
    else: # offline policy, we fall back to vanilla PG
        return vanila_pg(new_probs, loss)

#def compute_gae(rewards, values, gamma=0.99, tau=0.95):
def gae(rewards, values, gamma, tau):
    """ paper : https://arxiv.org/abs/1506.02438
        explained : https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/
        code : https://github.com/higgsfield/RL-Adventure-2 : ppo notebook
        + low variance + low bias
        - perf bottleneck to implement with replay buffer
    """

    value = 0
    if len(values) != len(rewards):
        value = gamma * values[-1]
        values = values[:-1]

    delta = rewards[-1] - values[-1] + value
    gae = delta

    returns = [gae + values[-1]]
    for step in reversed(range(len(rewards) - 1)):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * tau * gae
        returns.insert(0, gae)

    return returns

def td_lambda(rewards, n_step, gamma):
    """ paper : ...
        + low variance
        - high bias
    """
    return list(map(
        lambda i: sum(map(lambda t_r: t_r[1] * gamma**t_r[0], enumerate(rewards[i:i+n_step]))),
        range(len(rewards))))

def discount(rewards, gamma):
    """
    this we mostly dont using, unless we want to go for REINFORCE
    code : MSFT reinforcement learning explained course
    """
    discounted_r = np.zeros(len(rewards))
    running_add = 0.
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r
