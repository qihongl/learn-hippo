import numpy as np
import torch

from torch.nn.functional import smooth_l1_loss
from torch.distributions import Categorical

'''helpers'''

eps = np.finfo(np.float32).eps.item()


def compute_returns(rewards, gamma=0, normalize=False):
    """compute return in the standard policy gradient setting.

    Parameters
    ----------
    rewards : list, 1d array
        immediate reward at time t, for all t
    gamma : float, [0,1]
        temporal discount factor
    normalize : bool
        whether to normalize the return
        - default to false, because we care about absolute scales

    Returns
    -------
    1d torch.tensor
        the sequence of cumulative return

    """
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


def pick_action(action_distribution):
    """action selection by sampling from a multinomial.

    Parameters
    ----------
    action_distribution : 1d torch.tensor
        action distribution, pi(a|s)

    Returns
    -------
    torch.tensor(int), torch.tensor(float)
        sampled action, log_prob(sampled action)

    """
    m = Categorical(action_distribution)
    a_t = m.sample()
    log_prob_a_t = m.log_prob(a_t)
    return a_t, log_prob_a_t


def get_reward(a_t, a_t_targ, dk_id, penalty, allow_dk=True):
    """define the reward function at time t

    Parameters
    ----------
    a_t : int
        action
    a_t_targ : int
        target action
    dk_id : int
        the action id of the don't know unit
    penalty : int
        the penalty magnitude of making incorrect state prediction
    allow_dk : bool
        if True, then activating don't know makes r_t = 0, regardless of a_t

    Returns
    -------
    torch.FloatTensor, scalar
        immediate reward at time t

    """
    if a_t == dk_id and allow_dk:
        r_t = 0
    elif a_t_targ == a_t:
        r_t = 1
    else:
        r_t = -penalty
    return torch.tensor(r_t).type(torch.FloatTensor).data


def compute_a2c_loss(probs, values, returns):
    """compute the objective node for policy/value networks

    Parameters
    ----------
    probs : list
        action prob at time t
    values : list
        state value at time t
    returns : list
        return at time t

    Returns
    -------
    torch.tensor, torch.tensor
        Description of returned object.

    """
    policy_grads, value_losses = [], []
    for prob_t, v_t, R_t in zip(probs, values, returns):
        A_t = R_t - v_t.item()
        # A_t = R_t
        policy_grads.append(-prob_t * A_t)
        value_losses.append(
            smooth_l1_loss(torch.squeeze(v_t), torch.squeeze(R_t))
        )
    loss_policy = torch.stack(policy_grads).sum()
    loss_value = torch.stack(value_losses).sum()
    return loss_policy, loss_value
