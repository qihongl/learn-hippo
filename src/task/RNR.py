# import torch
import numpy as np
from utils.utils import to_pth
from task.TwilightZone import TwilightZone
# import matplotlib.pyplot as plts


class RNR():

    def __init__(
            self,
            n_param,
            n_branch,
            n_parts=3,
            n_timestep=None,
            p_rm_ob_enc=0,
            p_rm_ob_rcl=0,
            sampling_mode='enumerative'
    ):
        self.n_param = n_param
        self.n_branch = n_branch
        self.n_parts = n_parts
        self.p_rm_ob_enc = p_rm_ob_enc
        self.p_rm_ob_rcl = p_rm_ob_rcl
        # inferred params
        self.x_dim = (n_param * n_branch) * 2 + n_branch
        self.y_dim = n_branch
        if n_timestep is None:
            self.T_part = n_param
        else:
            self.T_part = n_timestep
        self.T_total = self.T_part * n_parts
        # build a 2-part movie sampler
        self.tz = TwilightZone(
            n_param=n_param,
            n_branch=n_branch,
            n_parts=2,
            n_timestep=None,
            p_rm_ob_enc=p_rm_ob_enc,
            p_rm_ob_rcl=p_rm_ob_enc,
            sampling_mode='enumerative'
        )

    def _class_config_validation(self):
        assert self.n_parts >= 3, 'n_parts >= 3'
