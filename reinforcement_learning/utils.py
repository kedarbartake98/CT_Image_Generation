import queue
import random
import socket
import time
import pickle
from multiprocessing import Process
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
from sklearn import preprocessing

import gym
import numpy as np
import pyglet

from scipy.ndimage import zoom
# from a2c.common.atari_wrappers import wrap_deepmind


# https://github.com/joschu/modular_rl/blob/master/modular_rl/running_stat.py
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape=()):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self._n >= 2:
            return self._S/(self._n - 1)
        else:
            return np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


# Based on SimpleImageViewer in OpenAI gym
class Im(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True

        assert arr.shape == (self.height, self.width), \
            "You passed in an image with the wrong number shape"

        image = pyglet.image.ImageData(self.width, self.height,
                                       'L', arr.tobytes(), pitch=-self.width)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


def get_port_range(start_port, n_ports, random_stagger=False):
    # If multiple runs try and call this function at the same time,
    # the function could return the same port range.
    # To guard against this, automatically offset the port range.
    if random_stagger:
        start_port += random.randint(0, 20) * n_ports

    free_range_found = False
    while not free_range_found:
        ports = []
        for port_n in range(n_ports):
            port = start_port + port_n
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                ports.append(port)
            except socket.error as e:
                if e.errno == 98 or e.errno == 48:
                    print("Warning: port {} already in use".format(port))
                    break
                else:
                    raise e
            finally:
                s.close()
        if len(ports) < n_ports:
            # The last port we tried was in use
            # Try again, starting from the next port
            start_port = port + 1
        else:
            free_range_found = True

    return ports


def profile_memory(log_path, pid):
    import memory_profiler
    def profile():
        with open(log_path, 'w') as f:
            # timeout=99999 is necessary because for external processes,
            # memory_usage otherwise defaults to only returning a single sample
            # Note that even with interval=1, because memory_profiler only
            # flushes every 50 lines, we still have to wait 50 seconds before
            # updates.
            memory_profiler.memory_usage(pid, stream=f,
                                         timeout=99999, interval=1)
    p = Process(target=profile, daemon=True)
    p.start()
    return p


def batch_iter(data, batch_size, shuffle=False):
    idxs = list(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)  # in-place

    start_idx = 0
    end_idx = 0
    while end_idx < len(data):
        end_idx = start_idx + batch_size
        if end_idx > len(data):
            end_idx = len(data)

        batch_idxs = idxs[start_idx:end_idx]
        batch = []
        for idx in batch_idxs:
            batch.append(data[idx])

        yield batch
        start_idx += batch_size

# def make_env(env_id, seed=0):
#     import ct_env
#     env = gym.make(env_id)
#     env.seed(seed)
#     return wrap_deepmind(env)

def get_first_grp_struct():
    ll = []
    for x in range(12):
        ll.append([x, x*2 + 12, x*2 + 13, x+1])
    ll[-1][-1] = 0
    co_in = []
    for i,z in enumerate(ll):
        if i == 0:
            co_in.extend(z)
        else:
            co_in.extend(z[1:])
    return co_in

def get_third_grp_struct():
    ll = []
    for x in range(3):
        ll.append([x, x*2 + 3, x*2 + 4, x+1])
    ll[-1][-1] = 0
    co_in = []
    for i,z in enumerate(ll):
        if i == 0:
            co_in.extend(z)
        else:
            co_in.extend(z[1:])
    return co_in
    
def get_second_grp_struct():
    ll = []
    for x in range(6):
        ll.append([x, x*2 + 6, x*2 + 7, x+1])
    ll[-1][-1] = 0
    co_in = []
    for i,z in enumerate(ll):
        if i == 0:
            co_in.extend(z)
        else:
            co_in.extend(z[1:])
    return co_in

def load_samples():
    samples_data = pickle.load(open("newgooddata.pkl","rb"))
    return samples_data

def get_normalized_organ_data(samples):
    scaled_orgs = []
    org_est = []
    orgs = [[],[],[],[],[],[]]
    
    for img in samples:
        for i,org in enumerate(img):
            orgs[i].append(np.array(org).tolist())
            
    for org in range(6):
        organ = np.array(orgs[0])
        org_nor_est = preprocessing.MinMaxScaler()
        org_scaled = org_nor_est.fit_transform(organ)
        scaled_orgs.append(org_scaled)
        org_est.append(org_nor_est)
        
    scaled_samples = []
    for i in range(len(scaled_orgs[0])):
        scaled_img = [scaled_orgs[org][i].tolist() for org in range(6)]
        scaled_samples.append(scaled_img)
    
    return scaled_samples, org_est
#     tor_nor_est = preprocessing.MinMaxScaler()
#     torso_scaled = tor_nor_est.fit_transform(torso)
#     return torso_scaled, tor_nor_est
    

def load_componets():
    f1 = open('data/pgnn/pgnn/BMDSXY_NODES_POS1.txt', "r")
    f2 = open('data/pgnn/pgnn/BMDSXY_NODES_POS2.txt', "r")
    f3 = open('data/pgnn/pgnn/BMDSXY_NODES_POS3.txt', "r")
    f4 = open('data/pgnn/pgnn/BMDSXY_NODES_POS4.txt', "r")
    f5 = open('data/pgnn/pgnn/BMDSXY_NODES_POS5.txt', "r")
    f = open('data/pgnn/pgnn/BMDSXY_NODES_POS.txt', "r")

    data = [{}, {}, {}, {}, {}, {}]
    datal = [[], [], [], [], [], []]
    for j, f_ in enumerate([f, f1, f2, f3, f4, f5]):
        for i,curve in enumerate(f_):
            data[j][i] = np.reshape(np.array([int(x) for x in curve.split()]), (-1,2))
            datal[j].append(np.array([float((int(x)-255.5)/255.5) for x in curve.split()]).tolist())
    
    coord_nvs = [20, 8, 8, 8, 4, 4] # 3, 3 TODO
    ests = []
    for org in range(6):
        estimator = PCA(n_components=coord_nvs[org], 
                        svd_solver='randomized').fit(np.asarray(datal[org]))
        ests.append(estimator)
        
    return ests

def vector_to_image(vector):
    co_in = get_first_grp_struct()
    co_in123 = get_second_grp_struct()
    co_in45 = get_third_grp_struct()
    ests = load_componets()
    coord_or = [co_in, co_in123, co_in123, co_in123, co_in45, co_in45]
    coord_nvs = [20, 8, 8, 8, 3, 3]
    offset = 0
    new_points = []
    try:
        for org in range(6):
            sample_o = vector[offset : offset + coord_nvs[org]]
            offset += coord_nvs[org]
            co_in_o = coord_or[org]
            curves_es_o = ests[org].mean_
            for i,val in enumerate(sample_o):
                curves_es_o = curves_es_o + ests[org].components_[i]*val
            curves_es_o = np.reshape((curves_es_o*255.5 + 255.5), (-1, 2)).astype(int).tolist()
            c = [curves_es_o[index][0] for index in co_in_o]
            d = [curves_es_o[index][1] for index in co_in_o]
            tck, _ = splprep([c, d], s=0.0, per=1)
            new_points_o = splev(np.linspace(0, 1, 1000), tck)
            new_points.append(new_points_o)
    except:
        return None
    
    gray_values = [230, 100, 50, 150, 200, 0]  # TODO: change values for normalization
    img = np.ones((512,512))*255
    for ind,organ in enumerate(new_points):  # Need to be made faster / may be graphs
        organ = np.array(organ).astype(int)
        img[organ[0],organ[1]] = gray_values[ind]
        img[organ[0]+1,organ[1]] = gray_values[ind]
        img[organ[0],organ[1]+1] = gray_values[ind]
        img[organ[0]+1,organ[1]+1] = gray_values[ind]
        img[organ[0]-1,organ[1]] = gray_values[ind]
        img[organ[0],organ[1]-1] = gray_values[ind]
        img[organ[0]-1,organ[1]-1] = gray_values[ind]
        
    return img
