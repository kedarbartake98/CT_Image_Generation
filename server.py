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

IMG_SOURCES = { 
    'source_img': 'static/images/source.png',
    'dest_img': 'static/images/dest.png',
    'interpolated': 'static/images/interpolated.png'
}

def clear_files():
    '''
    Cleanup function
    '''
    for key in IMG_SOURCES:

        filename = IMG_SOURCES[key]
        
        if os.path.exists(filename):
            os.remove(filename)

class Images:
    '''
    Object having functions to generate images from and interpolate 
    in the PCA space
    ''' 
    def __init__(self):
        self.source = None
        self.dest = None 
        self.interpolated = None

    def clear(self):
        self.source = None
        self.dest = None 
        self.interpolated = None

    def set_new_src_dest(self):
        self.clear()
        src_img, src_sample = pca.generate_random_image_sample()
        self.source = (src_img, src_sample)
        print('Writing')
        cv2.imwrite(IMG_SOURCES['source_img'], src_img)

        dst_img, dst_sample = pca.generate_random_image_sample()
        self.dest = (dst_img, dst_sample)
        cv2.imwrite(IMG_SOURCES['dest_img'], dst_img)       

    def interpolate(self, inc):

        self.interpolated = pca.interpolate(self.source[1], self.dest[1], inc)
        cv2.imwrite(IMG_SOURCES['interpolated'], self.interpolated)

app = Flask(__name__)
pca = PCA_rep(config)
image = Images()

@app.route('/')
def entrypoint():
    return render_template('index.html')

################### Code for Image generation/interpolation ####################

@app.route('/generate_images', methods=['POST'])

def generate():
    print('generate called')
    image.set_new_src_dest()
    return {'source' : url_for('static', filename='images/source.png'),
            'dest' : url_for('static', filename="images/dest.png"),
            'interpolated': url_for('static',  filename='images/source.png')}

@app.route('/interpolate', methods=['POST'])

def interpolate():
    inc = request.form['inc']
    # print('Inc', inc)
    image.interpolate(float(inc))
    return {'interpolated': url_for('static', \
                                    filename="images/interpolated.png")}

################### Code for sending segments to frontend ######################

@app.route('/get_segments', methods=['GET','POST'])

def get_segments():
    print("#"*100)
    print("Testing path pipe")
    print("#"*100)

    path_dict = path_pipe.get(block=True)
    folder = os.path.join('static/rl_sample_images', path_dict['random_folder'])

    print(path_dict['random_folder'])

    while len(os.listdir(folder))<11:
        pass

    filenames = ['img1', 'img2', 'img_mid']
    filenames += ['sample_{}'.format(i) for i in range(1,9)]

    for filename in filenames:
        path_dict[filename] = url_for('static',\
                                       filename=path_dict[filename])
    print(path_dict)
    return jsonify(path_dict)

################### Code for submitting preferences ############################

@app.route('/submit_prefs', methods=['POST'])

def submit_prefs():
    pref_dict = request.form['preferences']
    preferences = json.loads(pref_dict)
    print("Preferences",preferences)
    get_prefs_from_frontend(preferences, pref_pipe)
    return {"Done": "Done"}

############################## Backend Code #################################### 

def initialize_comms():
    '''
    Initialize global queues for communication between frontend and backend
    processes.
    '''
    seg_pipe = Queue(maxsize=8)
    pref_pipe = Queue(maxsize=8)
    start_policy_training_flag = Queue(maxsize=8)
    path_pipe = Queue(maxsize=1)

    return seg_pipe, pref_pipe, start_policy_training_flag, path_pipe

def start_backend(init_arg_tuple, comm_pipes):

    print('Starting Backend ...')

    # Unpack params
    general_params, pref_interface_params, \
    rew_pred_training_params, a2c_params = init_arg_tuple

    # Unpack pipes
    seg_pipe, pref_pipe, start_policy_training_flag, path_pipe = comm_pipes

    # Call main run function (3 modes from run.py)

    print("Calling the run function")
    print(a2c_params)

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
    clear_files()

    # Initialize communication pipes
    seg_pipe, pref_pipe, start_policy_training_flag, path_pipe = initialize_comms()
    comm_pipes = (seg_pipe, pref_pipe, start_policy_training_flag, path_pipe)

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
