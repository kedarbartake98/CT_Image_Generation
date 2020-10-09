import config
from graph import get_graph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
from scipy.interpolate import splprep, splev
from sklearn import decomposition
from sklearn.neighbors import NearestNeighbors

def read_data(config=config):

    pos_files, organ_names = config.POS_FILES, config.ORGANS
    coordinates = {}
    max_length = 0

    for i, organ in enumerate(organ_names):

        with open(pos_files[i], 'r') as f:
            print('Reading {}'.format(organ))
            lines = f.read().split('\n')
            lines = [line for line in lines if len(line.strip())!=0]
            max_length = max(len(lines), max_length)
            coordinates[organ] = lines

    # padding list with NaNs since unequal line numbers
    for organ in organ_names:
        coordinates[organ]+=[np.nan]*(max_length - len(coordinates[organ]))
    
    df = pd.DataFrame(coordinates)

    return df

class PCA_rep:

    def __init__(self, config):

        self.df = read_data(config)
        self.organs = config.ORGANS
        self.components = config.COMPONENTS
        self.co_in, self.co_in123, self.co_in45 = get_graph()
        self.get_graph_mapping()
        self.data_arrs = self.get_data_arrs()
        self.pca_est, self.transformed = self.get_estimators()
        self.cluster_torsos(neighbors=30)

    def get_graph_mapping(self):

        self.graph_mapping = {
            'Torso': self.co_in,
            'Left Lung':self.co_in123,
            'Right Lung':self.co_in123,
            'Heart':self.co_in123,
            'Spinal Cord':self.co_in45,
            'Oesophagus':self.co_in45
        }

    def get_data_arrs(self):

        data_arrs = {}

        for organ in self.organs:

            coord_list = self.df[organ].dropna().tolist()
            coord_list = [np.array([float((int(x)-255.5)/255.5) 
                         for x in curve.split()]) 
                         for curve in coord_list]
            data_arrs[organ] = coord_list

        return data_arrs

    def get_estimators(self):

        estimators = {}
        transformed = {}

        for organ in self.organs:

            estimator = PCA(n_components=self.components[organ],
                            svd_solver='randomized')
            estimator.fit(np.asarray(self.data_arrs[organ]))
            estimators[organ] = estimator
            transformed[organ] = estimator.transform(np.asarray(self.data_arrs[organ]))

        return estimators, transformed

    def cluster_torsos(self, neighbors=30):

        self.NNbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='auto')
        self.NNbrs.fit(self.transformed['Torso'])

    def generate_sample(self):

        # Generating random torso from PCA plane
        X_pca = self.transformed['Torso']
        selected_idx = np.random.permutation(X_pca.shape[0])[0]

        # Generate random torso 
        # sample from mean and covariance matrix of neighbors
        distances, indices = self.NNbrs.kneighbors(X_pca)
        covMatrix = np.cov(X_pca[indices[selected_idx]].T,bias=True)
        mean = np.mean(X_pca[indices[selected_idx]], axis=0)
        sample = np.random.multivariate_normal(mean, covMatrix)

        return self.generate_other_organs(sample)

    def generate_other_organs(self, torso_sample):

        # Generate nearest neighbors of the generated sample
        distances, indices = self.NNbrs.kneighbors(np.asarray([torso_sample,]))

        # Find corresponding instances of other organs for each neighbor
        data_indices = {}

        other_organs = self.organs.copy()
        other_organs.remove('Torso')

        for organ in other_organs:

            # Taking those indices for specific organs that have valid entries
            data_indices[organ] = [index for index in indices[0]
                                   if index<len(self.data_arrs[organ])]

        # if any organ gets no samples
        if any([data_indices[organ]==[] for organ in other_organs]):
            return self.generate_sample()

        means, covs, generated_sample = {}, {}, {}

        for organ in other_organs:

            X = self.transformed[organ]
            X = X[np.asarray(data_indices[organ])]

            covs[organ] = np.cov(X.T,bias=True)
            means[organ] = np.mean(X, axis=0)

        # Filling the generated sample
        generated_sample['Torso'] = torso_sample

        for organ in other_organs:

            generated_sample[organ] = np.random.multivariate_normal(means[organ], 
                                                                    covs[organ])

        return generated_sample


    def sample_to_curves(self, sample):

        sample_curves = {}

        for organ in self.organs:

            estimator = self.pca_est[organ]
            curve = estimator.mean_

            for i, val in enumerate(sample[organ]):

                curve = curve + estimator.components_[i]*val

            curve = np.reshape((curve*255.5 + 255.5), (-1, 2))

            sample_curves[organ] = process_curve(curve)

        return sample_curves

    def get_image(self, sample_curves, check_intersection=True):

        cur_img = np.zeros((512, 512), dtype="int32" )
        
        to_po, ll_po, rl_po, he_po, sp_po, eo_po = [],[],[],[],[],[]
        spl_lists = [to_po, ll_po, rl_po, he_po, sp_po, eo_po]

        for i, organ in enumerate(self.organs):

            int_x = sample_curves[organ]['int_x']
            int_y = sample_curves[organ]['int_y']
            
            c = [int_x[index] for index in self.graph_mapping[organ]]
            d = [int_y[index] for index in self.graph_mapping[organ]]

            try:

                tck, _ = splprep([c,d], s=0.0, per=1)
                new_points = splev(np.linspace(0,1,1000), tck)

                for x,y in zip(new_points[0], new_points[1]):

                    x, y = int(x), int(y)
                    spl_lists[i].append((x,y))
                    
                    cur_img[x,y] = cur_img[x+1,y] = cur_img[x-1,y] = \
                    cur_img[x,y+1] = cur_img[x,y-1] = cur_img[x+1,y+1] = \
                    cur_img[x-1,y-1] = cur_img[x+1,y-1] = cur_img[x-1,y+1] = \
                    cur_img[x+2,y] = cur_img[x-2,y] = cur_img[x,y+2] = \
                    cur_img[x,y-2] = cur_img[x+2,y+2] = cur_img[x-2,y-2] = \
                    cur_img[x+2,y-2] = cur_img[x-2,y+2] = 255
            except:
                return "ERROR"
        
        # Checking intersection
        if check_intersection:

            np_he_to = list(set(he_po) & set(to_po))
            np_he_ll = list(set(he_po) & set(ll_po))
            np_he_rl = list(set(he_po) & set(rl_po))
            np_he_sp = list(set(he_po) & set(sp_po))

            np_rl_to = list(set(rl_po) & set(to_po))
            np_rl_ll = list(set(rl_po) & set(ll_po))
            np_rl_sp = list(set(rl_po) & set(sp_po))

            np_ll_to = list(set(ll_po) & set(to_po))
            np_ll_sp = list(set(ll_po) & set(sp_po))


            if np_he_to or np_he_ll or np_he_rl or np_he_sp or np_rl_to\
               or np_rl_ll or np_rl_sp or np_ll_to or np_ll_sp:
                return "ERROR"

        return cur_img    

    def get_img_from_sample(self, sample):

        sample_curves = self.sample_to_curves(sample)

        img = self.get_image(sample_curves)

        return img

    def generate_random_image_sample(self):
        img = "ERROR"

        while isinstance(img, str):
            sample = self.generate_sample()
            img = self.get_img_from_sample(sample)
            print('Retrying ...')
        
        return img, sample

    def interpolate(self, sample1, sample2, inc):

        interp_sample = {}

        for organ in self.organs:

            interp_sample[organ] = (1-inc)*(sample1[organ])+inc*(sample2[organ])

        curves = self.sample_to_curves(interp_sample)

        return self.get_image(curves, check_intersection=False)

def process_curve(curve):
    
    coords = []
    int_x, int_y = [], []

    for x,y in curve:
        x = int(x)
        y = int(y)

        if x > 509:
            x = 509
        if x < 2:
            x = 2
        if y > 509:
            y = 509
        if y < 2:
            y = 2

        int_x.append(x)
        int_y.append(y)
        coords.append((x,y))

    curve_det = {'int_x':int_x,
                 'int_y':int_y,
                 'coords': coords}
    
    return curve_det
