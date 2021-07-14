#!/usr/bin/env python

"""
A simple CLI-based interface for querying the user about segment preferences.
"""

import logging
import queue
import time
from itertools import combinations
from multiprocessing import Queue
from random import shuffle
from reinforcement_learning.utils import vector_to_image

import easy_tf_log
import numpy as np
import string, os
import imageio

class PrefInterface:

    def __init__(self):
        self.seg_block = []

    def run(self, seg_pipe, pref_pipe, path_pipe, pref_db_pipe):

        while True:

            # Initialize segment block
            self.seg_block = []
            self.seg_pairs = set()
            # Collect 8 segments
            self.recv_segments(seg_pipe)

            # Process the segments and send them to the UI
            images, level = self.make_images()
            self.images_to_frontend(images, path_pipe, level)

            # Get the preferences from the UI
            pref_dict = self.get_prefs_from_frontend(pref_db_pipe)

            print('Pref_dict received:\n {}'.format(pref_dict))

            # Process segments and associate with user prefs
            self.process_prefs(pref_dict)

            # Process all possible pairs of segments and write them to db
            self.process_segment_pairs(pref_pipe)


    def recv_segments(self, seg_pipe):
        """
        Collect 8 segments for sending to the user from segment pipe
        """
        while len(self.seg_block)<8:

            try:
                segment = seg_pipe.get(block=True, timeout=2)
                self.seg_block.append(segment)

            except queue.Empty:
                pass

        return 

    def make_images(self):
        '''
        Process segments to make images from them
        '''        
        vector_img_1 = self.seg_block[0].frames[24][:50]
        vector_img_2 = self.seg_block[0].frames[24][50:100]
        vector_img_middle = [] 

        int_lvl = self.seg_block[0].frames[24][100]

        if int_lvl == 0:
            vector_img_middle = 0.7*vector_img_1 + 0.3*vector_img_2 
        elif int_lvl == 1:
            vector_img_middle = 0.5*vector_img_1 + 0.5*vector_img_2
        else:
            vector_img_middle = 0.3*vector_img_1 + 0.7*vector_img_2

        img1 = vector_to_image(vector_img_1)
        img2 = vector_to_image(vector_img_2)
        img_mid = vector_to_image(vector_img_middle)
        
        eight_images = []    
        for seg in self.seg_block:
            torso = seg.frames[24][101:]
            torso = np.append(torso, vector_img_middle[20:])
            eight_images.append(vector_to_image(torso))

        images = [img1, img2, img_mid]+eight_images

        return images, int_lvl

    def images_to_frontend(self, image_list, path_pipe, int_lvl):
        '''
        Create a random folder to save images and send
        the path mapping to frontend
        '''
        print('Sending path dict to front end')

        path_dict = self.get_path_dict(image_list, int_lvl)

        while True:
            try:
                path_pipe.put(path_dict, block=True)
                return 

            except queue.Full:
                time.sleep(1)


    def get_path_dict(self, images, int_lvl):

        # Create a random folder
        random_foldername = [np.random.choice([char for char in string.ascii_letters]) 
                             for k in range(0,12)]

        random_foldername = ''.join(random_foldername)
        folder_location = os.path.join('static/rl_sample_images', 
                                        random_foldername)
        os.makedirs(folder_location)

        # create filepaths 
        filenames = ['img1', 'img2', 'img_mid']
        filenames += ['sample_{}'.format(i) for i in range(1,9)]

        filepaths = [os.path.join(folder_location, filename+'.png')
                     for filename in filenames]

        mapping = [(images[i], filepaths[i]) for i in range(11)]

        filepaths = [filepath.split('static/')[1] for filepath in filepaths]

        image_listing = dict(zip(filenames, filepaths))
        image_listing['random_folder'] = random_foldername
        level_mapping = {0:0.3, 1:0.5, 2:0.7}
        image_listing['level'] = level_mapping[int_lvl]*100

        _ = [imageio.imwrite(filepath, image) 
             for (image, filepath) in mapping]

        return image_listing


    def get_prefs_from_frontend(self, pref_db_pipe):
        '''
        Get the prefs dict from frontend
        '''

        while True:

            try:
                pref_dict = pref_db_pipe.get(block=True, timeout=1)
                return pref_dict

            except queue.Empty:
                time.sleep(2)

    def process_prefs(self, pref_dict):
        '''
        Process prefs received from user
        '''
        pass
    
    def process_segment_pairs(self, pref_pipe):
        '''
        Process all combinations of segment pairs and write them to db
        '''

        segment_idxs = list(range(len(self.seg_block)))
        possible_pairs = combinations(segment_idxs, 2)

        for i1, i2 in possible_pairs:

            if not (i1,i2) in self.seg_pairs: 
                
                self.seg_pairs.add((i1,i2))
                self.seg_pairs.add((i2,i1))
                
                s1, s2 = self.seg_block[i1], self.seg_block[i2]
                ### write code for identifying a pref for s1, s2
                if (s1.better_than_linear and not s2.better_than_linear) or \
                    (not s1.worse_torso and s2.worse_torso):
                    print('$'*100)
                    print('Setting 1-0 Pref')
                    pref = (1.0, 0.0)
                elif (not s1.better_than_linear and s2.better_than_linear) or \
                    (s1.worse_torso and not s2.worse_torso):
                    print('$'*100)
                    print('Setting 0-1 Pref')
                    pref = (0.0, 1.0)
                else:
                    print('$'*100)
                    print('Setting Default Pref')
                    pref = (0.5, 0.5)

                if pref is not None:
                    # We don't need the rewards from this point on, so just send
                    # the frames  ###? because the rewards are known by reward classifier???
                    while True:
                        try:
                            pref_pipe.put((s1.frames, s2.frames, pref))
                            break
                        except queue.Full:
                            time.sleep(0.5)
                            continue

            else:
                print('Prev pair')