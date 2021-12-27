from bidict import bidict

ALL_SUBDIRS = ['ckpts', 'data', 'figs']

# file name templates
CKPT_TEMPLATE = 'ckpt_ep-%d.pt'
CACHE_FNAME = 'testing_info.pkl'

ENV_JSON_FNAME = 'params_env.json'
NET_JSON_FNAME = 'params_net.json'

#
ALL_ENC_MODE = ['cum', 'disj']

# conditions
TZ_COND_DICT = bidict({0: 'RM', 1: 'DM', 2: 'NM'})
P_TZ_CONDS = [.25, .25, .5]
TZ_CONDS = list(TZ_COND_DICT.values())
