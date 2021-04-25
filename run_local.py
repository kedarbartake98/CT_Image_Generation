import logging
import os
from os import path as osp
import sys
import time
from multiprocessing import Process, Queue
import cloudpickle
import easy_tf_log

from reinforcement_learning.ct_env_n import CustomEnv
# from a2c.common import set_global_seeds
# from a2c.common.vec_env.subproc_vec_env import SubprocVecEnv
from reinforcement_learning.pref_db import PrefDB, PrefBuffer
from reinforcement_learning.pref_interface import PrefInterface
from reinforcement_learning.reward_predictor import RewardPredictorEnsemble
from reinforcement_learning.reward_predictor_core_network import net_cnn
from reinforcement_learning.utils import get_port_range
from reinforcement_learning.params import parse_args, PREFS_VAL_FRACTION
from reinforcement_learning.a2c.a2c.a2c import learn
from reinforcement_learning.a2c.a2c.policies import MlpPolicy
from rl_init_params import init_arg_tuple



policy_fn = MlpPolicy
reward_predictor_network = net_cnn

general_params, pref_interface_params, \
    rew_pred_training_params, a2c_params = init_arg_tuple

env = CustomEnv(a2c_params['n_envs'])

# if general_params['debug']:
#     logging.getLogger().setLevel(logging.DEBUG)

def make_reward_predictor():
    return RewardPredictorEnsemble(
#         log_dir=general_params['log_dir'],
            batchnorm=True,
            dropout=False,
        lr=rew_pred_training_params['lr'],
        core_network=reward_predictor_network)

# ckpt_dir = osp.join(log_dir, 'policy_checkpoints')

# if not os.path.exists(ckpt_dir):
#     os.makedirs(ckpt_dir)

reward_predictor = make_reward_predictor()
# misc_logs_dir = osp.join(log_dir, 'a2c_misc')
# easy_tf_log.set_dir(misc_logs_dir)
learn(
    policy=policy_fn,
    env=env,
    reward_predictor=reward_predictor, **a2c_params)


