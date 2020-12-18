'''
Define initial parameters as tuple of dicts, refer 
reinforcement_learning/params.py

-- init_arg_tuple is the variable

# TODO Arjun
'''

general_args = {
	'mode': 'gather_initial_prefs',
	'log_dir': 'logs'
}

pref_interface_args = {
	'param' : 'param'
}

reward_predictor_args = {
	'param' : 'param'
}

a2c_args = {
    'log_interval': 100, 
    'ent_coef': 0.01,
    'n_envs': 8,
    'seed': 0,
    "lr_zero_million_timesteps": None,
    'lr': 7e-4,
    'policy_ckpt_interval': 100,
    'million_timesteps': 10
}

init_arg_tuple = (general_args, pref_interface_args, 
				  reward_predictor_args, a2c_args)