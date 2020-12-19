'''
Define initial parameters as tuple of dicts, refer 
reinforcement_learning/params.py

-- init_arg_tuple is the variable

# TODO Arjun
'''
from reinforcement_learning.utils import Scheduler

general_args = {
	'mode': 'gather_initial_prefs',
	'log_dir': 'logs',
	'max_prefs': 20000,
	'n_initial_prefs': 8
}

pref_interface_args = {
	'param' : 'param'
}

reward_predictor_args = {
	'lr' : 2e-4
}

a2c_args = {
    'log_interval': 100, 
    'ent_coef': 0.01,
    'n_envs': 8,
    'lr_scheduler': Scheduler(v=2e-4, nvalues=1e6, schedule='linear')
}

init_arg_tuple = (general_args, pref_interface_args, 
				  reward_predictor_args, a2c_args)