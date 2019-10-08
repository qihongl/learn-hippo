import numpy as np
import torch

from torch.nn.functional import smooth_l1_loss

'''helpers'''

eps = np.finfo(np.float32).eps.item()


def get_reward(a_t, y_t, penalty, allow_dk=True):
    """define the reward function at time t

    Parameters
    ----------
    a_t : int
        action
    a_t_targ : int
        target action
    penalty : int
        the penalty magnitude of making incorrect state prediction
    allow_dk : bool
        if True, then activating don't know makes r_t = 0, regardless of a_t

    Returns
    -------
    torch.FloatTensor, scalar
        immediate reward at time t

    """
    dk_id = y_t.size()[0]
    # if y_t is all zeros (delay period), then action target DNE
    if torch.all(y_t == 0):
        # -1 is not in the range of a_t, so r_t = penalty unless a_t == dk
        a_t_targ = torch.tensor(-1)
    else:
        a_t_targ = torch.argmax(y_t)
    # compare action vs. target action
    if a_t == dk_id and allow_dk:
        r_t = 0
    elif a_t_targ == a_t:
        r_t = 1
    else:
        r_t = - penalty
    return torch.from_numpy(np.array(r_t)).type(torch.FloatTensor).data
    # return torch.tensor(r_t).type(torch.FloatTensor).clone().detach()


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
    # compute cumulative discounted reward since t, for all t
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # normalize w.r.t to the statistics of this trajectory
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


def compute_a2c_loss(probs, values, returns, use_V=True):
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
        if use_V:
            A_t = R_t - v_t.item()
            value_losses.append(
                smooth_l1_loss(torch.squeeze(v_t), torch.squeeze(R_t))
            )
        else:
            A_t = R_t
            value_losses.append(torch.FloatTensor(0).data)
        # accumulate policy gradient
        policy_grads.append(-prob_t * A_t)
    policy_gradient = torch.stack(policy_grads).sum()
    value_loss = torch.stack(value_losses).sum()
    return policy_gradient, value_loss
