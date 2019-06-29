import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models.initializer import ortho_init


class A2C(nn.Module):
    """a MLP actor-critic network
    process: relu(Wx) -> pi, v

    Parameters
    ----------
    dim_input : int
        dim state space
    dim_hidden : int
        number of hidden units
    dim_output : int
        dim action space

    Attributes
    ----------
    ih : torch.nn.Linear
        input to hidden mapping
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network
    _init_weights : helper func
        default weight init scheme

    """

    def __init__(self, dim_input, dim_hidden, dim_output):
        super(A2C, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.ih = nn.Linear(dim_input, dim_hidden)
        self.actor = nn.Linear(dim_hidden, dim_output)
        self.critic = nn.Linear(dim_hidden, 1)
        ortho_init(self)

    def forward(self, x, beta=1, return_h=False):
        """compute action distribution and value estimate, pi(a|s), v(s)

        Parameters
        ----------
        x : a vector
            a vector, state representation
        beta : float, >0
            softmax temp, big value -> more "randomness"

        Returns
        -------
        vector, scalar
            pi(a|s), v(s)

        """
        h = F.relu(self.ih(x))
        action_distribution = _softmax(self.actor(h), beta)
        value_estimate = self.critic(h)
        # pack output
        outputs_ = [action_distribution, value_estimate]
        if return_h:
            outputs_.append(torch.squeeze(h))
        return outputs_

    def pick_action(self, action_distribution):
        a_t, log_prob_a_t = _pick_action(action_distribution)
        return a_t, log_prob_a_t


class A2C_linear(nn.Module):
    """a linear actor-critic network
    process: x -> pi, v

    Parameters
    ----------
    dim_input : int
        dim state space
    dim_output : int
        dim action space

    Attributes
    ----------
    actor : torch.nn.Linear
        the actor network
    critic : torch.nn.Linear
        the critic network

    """

    def __init__(self, dim_input, dim_output):
        super(A2C_linear, self).__init__()
        self.actor = nn.Linear(dim_input, dim_output)
        self.critic = nn.Linear(dim_input, 1)
        ortho_init(self)

    def forward(self, x, beta=1):
        """compute action distribution and value estimate, pi(a|s), v(s)

        Parameters
        ----------
        x : a vector
            a vector, state representation
        beta : float, >0
            softmax temp, big value -> more "randomness"

        Returns
        -------
        vector, scalar
            pi(a|s), v(s)

        """
        action_distribution = _softmax(self.actor(x), beta)
        value_estimate = self.critic(x)
        return action_distribution, value_estimate

    def pick_action(self, action_distribution):
        a_t, log_prob_a_t = _pick_action(action_distribution)
        return a_t, log_prob_a_t


def _softmax(z, beta):
    """helper function, softmax with beta

    Parameters
    ----------
    z : torch tensor, has 1d underlying structure after torch.squeeze
        the raw logits
    beta : float, >0
        softmax temp, big value -> more "randomness"

    Returns
    -------
    1d torch tensor
        a probability distribution | beta

    """
    assert beta > 0
    # softmax the input to a valid PMF
    pi_a = F.softmax(torch.squeeze(z / beta), dim=0)
    # make sure the output is valid
    if torch.any(torch.isnan(pi_a)):
        raise ValueError(f'Softmax produced nan: {z} -> {pi_a}')
    return pi_a


def _pick_action(action_distribution):
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
