import os
import torch
import json
import pickle

from copy import deepcopy
from utils.utils import vprint
from utils.constants import CKPT_TEMPLATE, ALL_SUBDIRS, NET_JSON_FNAME

"""helper func, ckpt io
"""


def build_log_path(
        subj_id, p,
        log_root=None,
        verbose=True
):
    # create dir names
    exp_str = f'{p.env.exp_name}'
    graph_str = f'p-{p.env.n_param}_b-{p.env.n_branch}'
    # if p.env.pad_len > 0 or p.env.pad_len == 'random':
    graph_str += f'_pad-{p.env.pad_len}'
    prob_str = 'tp-%.2f' % (p.env.def_prob)
    penalty_str = f'lp-{p.env.penalty}'
    obs_str = 'p_rm_ob_rcl-%.2f_enc-%.2f' % (
        p.env.p_rm_ob_rcl, p.env.p_rm_ob_enc)
    # net params
    enc_str = f'enc-{p.net.enc_mode}_size-{p.net.enc_size}'
    enc_capc_str = f'nmem-{p.net.n_mem}'
    recall_str = f'rp-{p.net.recall_func}_metric-{p.net.kernel}'
    network_str = f'h-{p.net.n_hidden}'
    train_str = f'lr-{p.net.lr}-eta-{p.net.eta}'
    curic_str = f'sup_epoch-{p.misc.sup_epoch}'
    subj_str = f'subj-{subj_id}'
    # compute the path
    log_path = os.path.join(
        log_root,
        exp_str, graph_str, prob_str, obs_str, penalty_str,
        enc_str, enc_capc_str, recall_str, network_str, train_str, curic_str,
        subj_str
    )
    log_subpath = {
        subdir: os.path.join(log_path, subdir) for subdir in ALL_SUBDIRS
    }
    # add rnr subpaths
    # if p.env.rnr.n_mvs is not None:
    #     log_subpath = update_rnr_log_subpath(log_subpath, p.env.rnr.n_mvs)
    # make subdirs for ckpts, activations, figures
    _make_all_dirs(log_path, log_subpath, verbose)
    return log_path, log_subpath


def update_rnr_log_subpath(log_subpath, n_mvs_rnr):
    """Add subpath for the rnr experiments

    e.g.
    log_subpath = update_rnr_log_subpath(log_subpath, p.env.rnr.n_mvs)

    Parameters
    ----------
    log_subpath : list
        see build_log_path()
    n_mvs_rnr : int
        see p.env.rnr.n_mvs

    Returns
    -------
    list
        `log_subpath` with rnr sub dirs

    """
    log_subpath['rnr-data'] = os.path.join(
        log_subpath['data'], 'n_mvs-%s' % n_mvs_rnr)
    log_subpath['rnr-ckpts'] = os.path.join(
        log_subpath['ckpts'], 'n_mvs-%s' % n_mvs_rnr)
    log_subpath['rnr-figs'] = os.path.join(
        log_subpath['figs'], 'n_mvs-%s' % n_mvs_rnr)
    return log_subpath


def _make_all_dirs(log_path, log_subpath, verbose):
    # create output dir
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
        vprint(verbose, f'Dir created: \n{log_path}')
    else:
        vprint(verbose, f'Use exisiting dir: \n{log_path}')

    for k, log_subpath_ in log_subpath.items():
        if not os.path.exists(log_subpath_):
            os.makedirs(log_subpath_, exist_ok=True)
            vprint(verbose, f'- sub dir: {k}')
        else:
            vprint(verbose, f'- use exisiting sub dir: {k}')


def save_ckpt(
        cur_epoch, log_path, agent, optimizer,
        ckpt_template=CKPT_TEMPLATE
):
    # compute fname
    ckpt_fname = ckpt_template % cur_epoch
    log_fpath = os.path.join(log_path, ckpt_fname)
    torch.save({
        'network_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, log_fpath)


def load_ckpt(
    epoch_load, log_path, agent,
    optimizer=None,
    ckpt_template=CKPT_TEMPLATE
):
    # compute fname
    ckpt_fname = ckpt_template % epoch_load
    log_fpath = os.path.join(log_path, ckpt_fname)
    # load the ckpt back
    checkpoint = torch.load(log_fpath)
    # unpack results
    agent.load_state_dict(checkpoint['network_state_dict'])
    if optimizer is None:
        optimizer = torch.optim.Adam(agent.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #
    agent.train()
    # msg
    print(f'epoch {epoch_load} loaded')
    return agent, optimizer


def save_all_params(datapath, params, args=None):
    msg = f'''Write experiment params and metadata to...
    {datapath}
    '''
    print(msg)

    # reformat
    env = deepcopy(params.env)
    env.def_path = env.def_path.tolist()
    # save input args
    if args is not None:
        with open(os.path.join(datapath, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    # save env params
    # env_outpath = os.path.join(datapath, ENV_JSON_FNAME)
    # assert os.path.exists(os.path.dirname(env_outpath))
    # with open(env_outpath, 'w') as f:
    #     json.dump(env.__dict__, f, indent=2)
    # save model params
    mod_outpath = os.path.join(datapath, NET_JSON_FNAME)
    assert os.path.exists(os.path.dirname(mod_outpath))
    with open(mod_outpath, 'w') as f:
        json.dump(params.net.__dict__, f, indent=2)


def pickle_save_dict(input_dict, save_path):
    """Save the dictionary

    Parameters
    ----------
    input_dict : type
        Description of parameter `input_dict`.
    save_path : type
        Description of parameter `save_path`.

    """
    pickle.dump(input_dict, open(save_path, "wb"))


def pickle_load_dict(fpath):
    """load the dict

    Parameters
    ----------
    fpath : type
        Description of parameter `fpath`.

    Returns
    -------
    type
        Description of returned object.

    """
    return pickle.load(open(fpath, "rb"))


def pickle_save_df(input_df, save_path):
    """Save panda dataframe.

    Parameters
    ----------
    input_df : type
        Description of parameter `input_df`.
    save_path : type
        Description of parameter `save_path`.

    """
    input_df.to_pickle(save_path)
