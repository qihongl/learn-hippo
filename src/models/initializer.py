import torch.nn as nn


def initialize_weights(agent, option='ortho'):
    if option == 'ortho':
        ortho_init(agent)
    elif option == 'xaiver_uniform':
        xavier_uniform_init(agent)
    else:
        raise ValueError(f'unrecognizable weight init scheme {option}')


def ortho_init(agent):
    """ortho init agent

    Parameters
    ----------
    agent : a pytorch agent
        Description of parameter `agent`.

    """
    for name, wts in agent.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(wts)
        elif 'bias' in name:
            nn.init.constant_(wts, 0)


def set_forget_gate_bias(lstm):
    # Set LSTM forget gate bias to 1
    for name, wts in lstm.named_parameters():
        if 'bias' in name:
            n = wts.size(0)
            forget_start_idx, forget_end_idx = n // 4, n // 2
            nn.init.constant_(wts[forget_start_idx:forget_end_idx], 1)


def xavier_uniform_init(agent):
    for wts in agent.parameters():
        if wts.data.ndimension() > 1:
            nn.init.xavier_uniform_(wts.data)
        else:
            nn.init.zeros_(wts.data)
