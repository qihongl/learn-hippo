# LOG_ROOT = '../log/'
# FIG_ROOT = '../fig/'
ALL_SUBDIRS = ['ckpts', 'data', 'figs']

# file name templates
CKPT_TEMPLATE = 'ckpt_ep-%d.pt'
CACHE_FNAME = 'testing_info.pkl'

ENV_JSON_FNAME = 'params_env.json'
NET_JSON_FNAME = 'params_net.json'

#
ALL_ENC_MODE = ['cum', 'disj']

#
# RNR_CKPT_TEMPLATE = 'rnr_ep-%d_tz-%d.pt'
# RNR_EVAL_FTEMPLATE = 'eval-rnr-e%d-nmvs%d.pkl'


# conditions
TZ_CONDS = ['RM', 'DM', 'NM']
RNR_CONDS = ['R', 'NR']
RNR_COND_DICT = {0: 'R', 1: 'NR'}

# sub condition parameters
OBS_RM_MODE = ['all', 'partial']


def rnr_log_fnames(epoch_load, n_epoch):
    """Short summary.

    e.g.
    ckpt_template, save_data_fname = rnr_log_fnames(epoch_load,n_epoch)

    Parameters
    ----------
    epoch_load : type
        Description of parameter `epoch_load`.
    n_epoch : type
        Description of parameter `n_epoch`.

    Returns
    -------
    type
        Description of returned object.

    """
    log_fstr = f'ep-{epoch_load}-%d'
    ckpt_template = 'ckpt_' + log_fstr + '.pt'
    save_data_fname = 'eval_' + log_fstr % (n_epoch) + '.pkl'
    return ckpt_template, save_data_fname
