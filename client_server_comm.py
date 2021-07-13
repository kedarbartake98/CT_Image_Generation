'''
Contains Utility functions for communicating between frontend and backend

'''

from multiprocessing import Pipe, Queue
from flask import Flask, render_template, url_for, jsonify, request
import pandas as pd
import os
import random, json
import config
import sys
import numpy as np
import cv2

def get_prefs_from_frontend(pref_dicts, pref_pipe, pref_db_pipe):
    '''
    Process dict/list of dicts received from frontend and put it in relevant 
    pipes/queues according to need
    '''
    # Put dict in pref_pipe
    print('CALLED prefs fromn frontend function')
    print(pref_dicts)
    pass