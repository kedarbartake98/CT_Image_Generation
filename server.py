from flask import Flask, render_template, url_for, jsonify, request
import pandas as pd
import os
import random, json
import config
import sys
import numpy as np
from PCA_rep import PCA_rep
from multiprocessing import Process, Queue
import cv2

from reinforcement_learning import run, a2c
from rl_init_params import init_arg_tuple

from client_server_comm import *

app = Flask(__name__)

@app.route('/')
def entrypoint():
    return render_template('index.html')

################### Code for sending segments to frontend ######################

@app.route('/get_segments', methods=['GET','POST'])

def get_segments():

    path_dict = path_pipe.get(block=True)
    folder = os.path.join('static/rl_sample_images', path_dict['random_folder'])

    while len(os.listdir(folder))<11:
        pass

    filenames = ['img1', 'img2', 'img_mid']
    filenames += ['sample_{}'.format(i) for i in range(1,9)]

    for filename in filenames:
        path_dict[filename] = url_for('static',\
                                       filename=path_dict[filename])
    return jsonify(path_dict)

################### Code for submitting preferences ############################

@app.route('/submit_prefs', methods=['POST'])

def submit_prefs():
    pref_dict = request.form['preferences']
    preferences = json.loads(pref_dict)
    get_prefs_from_frontend(preferences, pref_pipe, pref_db_pipe)
    return {"Done": "Done"}

############################## Backend Code #################################### 

def initialize_comms():
    '''
    Initialize global queues for communication between frontend and backend
    processes.
    '''
    seg_pipe = Queue(maxsize=8)
    pref_pipe = Queue(maxsize=8)
    path_pipe = Queue(maxsize=8)
    pref_db_pipe = Queue(maxsize=8)
    start_policy_training_flag = Queue(maxsize=8)

    return seg_pipe, pref_pipe, start_policy_training_flag, \
           path_pipe, pref_db_pipe

def start_backend(init_arg_tuple, comm_pipes):

    # Unpack params
    general_params, pref_interface_params, \
    rew_pred_training_params, a2c_params = init_arg_tuple

    # Unpack pipes
    seg_pipe, pref_pipe, start_policy_training_flag, \
    path_pipe, pref_db_pipe = comm_pipes

    run(general_params, a2c_params, pref_interface_params, 
        rew_pred_training_params, seg_pipe, pref_pipe, path_pipe, 
        start_policy_training_flag)

################################################################################

def run_web_app():
    app.run(debug=True)

if __name__=='__main__':
    '''
    The function will run the front end and backend in separate processes and 
    establish communication between them using multiprocessing Queues.
    '''

    # Initialize communication pipes
    seg_pipe, pref_pipe, start_policy_training_flag,\
    path_pipe, pref_db_pipe = initialize_comms()
    comm_pipes = (seg_pipe, pref_pipe, start_policy_training_flag, path_pipe,\
                  pref_db_pipe)

    # Calling the Reinforcement Learning Script
    print('Starting Backend ..')
    backend_process = Process(target=start_backend, args=(init_arg_tuple,
                                                          comm_pipes))
    backend_process.start()

    # Initializing Frontend
    print('Rendering Web App ...')
    fr_end_process = Process(target=run_web_app)
    fr_end_process.start()


    fr_end_process.join()
    backend_process.join()
