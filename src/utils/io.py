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
        mkdir=True,
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
    enc_capc_str = f'nmem-{p.n_event_remember}'
    recall_str = f'rp-{p.net.recall_func}_metric-{p.net.kernel}_cmpt-{p.net.cmpt}'
    network_str = f'h-{p.net.n_hidden}_hdec-{p.net.n_hidden_dec}'
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
    _make_all_dirs(log_path, log_subpath, mkdir=mkdir, verbose=verbose)
    return log_path, log_subpath


def _make_all_dirs(log_path, log_subpath, mkdir=True, verbose=False):
    # create output dir
    if not os.path.exists(log_path) and mkdir:
        os.makedirs(log_path, exist_ok=True)
        vprint(verbose, f'Dir created: \n{log_path}')
    else:
        vprint(verbose, f'Use exisiting dir: \n{log_path}')

    for k, log_subpath_ in log_subpath.items():
        if not os.path.exists(log_subpath_) and mkdir:
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
    if os.path.exists(log_fpath):
        # load the ckpt back
        checkpoint = torch.load(log_fpath)
        # unpack results
        agent.load_state_dict(checkpoint['network_state_dict'])
        if optimizer is None:
            optimizer = torch.optim.Adam(agent.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.train()
        # msg
        print(f'network weights - epoch {epoch_load} loaded')
        return agent, optimizer
    return None, None


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
    env_outpath = os.path.join(datapath, 'env.json')
    assert os.path.exists(os.path.dirname(env_outpath))
    with open(env_outpath, 'w') as f:
        json.dump(env.__dict__, f, indent=2)
    # save model params
    mod_outpath = os.path.join(datapath, NET_JSON_FNAME)
    assert os.path.exists(os.path.dirname(mod_outpath))
    with open(mod_outpath, 'w') as f:
        json.dump(params.net.__dict__, f, indent=2)


def get_test_data_dir(log_subpath, epoch_load, test_params):
    [fix_penalty, pad_len_test, slience_recall_time] = test_params

    if slience_recall_time is None:
        str_info = 'srt-None'
    else:
        str_info = f'srt-{slience_recall_time[0]}-{slience_recall_time[-1]}'

    test_data_subdir = os.path.join(
        f'epoch-{epoch_load}',
        f'penalty-{fix_penalty}',
        f'delay-{pad_len_test}',
        str_info
    )

    test_data_dir = os.path.join(log_subpath['data'], test_data_subdir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    return test_data_dir, test_data_subdir


def get_test_data_fname(n_examples_test, fix_cond=None, scramble=False):
    test_data_fname = f'n{n_examples_test}.pkl'
    if fix_cond is not None:
        test_data_fname = fix_cond + '-' + test_data_fname
    if scramble:
        test_data_fname = 'scramble' + '-' + test_data_fname
    return test_data_fname


def pickle_save_dict(input_dict, save_path):
    """Save the dictionary

    Parameters
    ----------
    input_dict : dict
        a dictionary to be saved
    save_path : str
        file path

    """
    with open(save_path, 'wb') as handle:
        pickle.dump(input_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_dict(fpath):
    """load the dict

    Parameters
    ----------
    fpath : str
        file path

    Returns
    -------
    type
        dict

    """
    return pickle.load(open(fpath, "rb"))


def pickle_save_df(input_df, save_path):
    """Save panda dataframe.

    Parameters
    ----------
    input_df : df
        a df to be saved
    save_path : str
        file path

    """
    input_df.to_pickle(save_path)


def load_env_metadata(log_subpath):
    env_data_path = os.path.join(log_subpath['data'], 'env.json')
    # test if the file exists
    if not os.path.exists(log_subpath['data']):
        print(log_subpath['data'])
        raise ValueError('Data path not found')
    if not os.path.isfile(env_data_path):
        raise ValueError(f'File: env.json not found')
    # load
    with open(env_data_path, 'r') as f:
        env_data = json.load(f)
    return env_data
