import os
import numpy as np
import tensorflow as tf
from collections import deque
from scipy.interpolate import splprep, splev

def sample(logits):
    noise1 = tf.compat.v1.random_uniform(tf.shape(logits))
#     noise2 = tf.random_uniform(tf.shape(logits))
    # the values should be random in the beginning, so should be different
    # in the beginning before returning usually the same actions for both
    # what if it chooses the 'do-nothing action?'
    # maybe just use sequence of actions as different output (8 actions = 8 output)
    # maybe don't use this framework at all and do the alternate classifier training ( seperate 
    # usual cnns) and reinforcement learning.
    return tf.compat.v1.argmax(logits - tf.compat.v1.log(-tf.compat.v1.log(3*noise1)), 1)#,
#             tf.argmax(logits - tf.log(-tf.log(noise2)), 1)


def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.compat.v1.variable_scope(scope):
        nin = x.get_shape()[1]
        w = tf.compat.v1.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.compat.v1.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w)+b
        h = act(z)
        return h
    
def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.compat.v1.log(z0) - a0), 1)

# def cat_entropy_softmax(p0):
#     return - tf.reduce_sum(p0 * tf.log(p0 + 1e-6), axis = 1)

def mse(pred, target):
    return tf.square(pred-target)/2.

def find_trainable_variables(key):
    with tf.compat.v1.variable_scope(key):
        return tf.compat.v1.trainable_variables()

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

def make_path(f):
    return os.makedirs(f, exist_ok=True)

def constant(p):
    return 1

def linear(p):
    return 1-p

def my_explained_variance(qpred, q):
    _, vary = tf.nn.moments(q, axes=[0, 1])
    _, varpred = tf.nn.moments(q - qpred, axes=[0, 1])
    check_shape([vary, varpred], [[]] * 2)
    return 1.0 - (varpred / vary)

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

def load_componets():
    f1 = open('/Users/arjunkrishna/CT_image_pca_visualization/data/pgnn/pgnn/BMDSXY_NODES_POS1.txt', "r")
    f2 = open('/Users/arjunkrishna/CT_image_pca_visualization/data/pgnn/pgnn/BMDSXY_NODES_POS2.txt', "r")
    f3 = open('/Users/arjunkrishna/CT_image_pca_visualization/data/pgnn/pgnn/BMDSXY_NODES_POS3.txt', "r")
    f4 = open('/Users/arjunkrishna/CT_image_pca_visualization/data/pgnn/pgnn/BMDSXY_NODES_POS4.txt', "r")
    f5 = open('/Users/arjunkrishna/CT_image_pca_visualization/data/pgnn/pgnn/BMDSXY_NODES_POS5.txt', "r")
    f = open('/Users/arjunkrishna/CT_image_pca_visualization/data/pgnn/pgnn/BMDSXY_NODES_POS.txt', "r")

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

def vectors_to_images(vectors):
    co_in = get_first_grp_struct()
    co_in123 = get_second_grp_struct()
    co_in45 = get_third_grp_struct()
    ests = load_componets()
    coord_or = [co_in, co_in123, co_in123, co_in123, co_in45, co_in45]
    vector0 = vectors[0]
    int_lvl = vector0[100]
    coord_nvs = [20, 8, 8, 8, 3, 3]
    offset = 0
    new_points = []
    rew = np.ones((len(vectors),) )
    try:
        for org in range(6):
            if org == 0:
                offset += coord_nvs[org]
                continue
            sample_o1 = vector0[offset : offset + coord_nvs[org]]
            sample_o2 = vector0[offset + 50: offset + coord_nvs[org]]
            if int_level == 0:
                sample_o = 0.7*sample_o1 + 0.3*sample_o2
            elif int_level == 1:
                sample_o = 0.5*sample_o1 + 0.5*sample_o2
            else:
                sample_o = 0.3*sample_o1 + 0.7*sample_o2
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
        return None, rew
    
    gray_values = [100, 50, 150, 200, 0]  # TODO: change values for normalization
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
        
    img = np.expand_dims(img, axis=0).repeat(len(vectors), 0)
    c = -1
    for vector, im in zip(vectors, img):
        c+=1
        torso = vector[101:]
        curves_es_o = ests[0].mean_
        for i,val in enumerate(torso):
            curves_es_o = curves_es_o + ests[0].components_[i]*val
        curves_es_o = np.reshape((curves_es_o*255.5 + 255.5), (-1, 2)).astype(int).tolist()
        try:
            c = [curves_es_o[index][0] for index in co_in]
            d = [curves_es_o[index][1] for index in co_in]
            tck, _ = splprep([c, d], s=0.0, per=1)
            new_points_o = splev(np.linspace(0, 1, 1000), tck)
            new_points_o = np.array(new_points_o).astype(int)
        except:
            rew[c] = 0
            continue
        im[new_points_o[0],new_points_o[1]] = 230
        im[new_points_o[0]+1,new_points_o[1]] = 230
        im[new_points_o[0],new_points_o[1]+1] = 230
        im[new_points_o[0]+1,new_points_o[1]+1] = 230
        im[new_points_o[0]-1,new_points_o[1]] = 230
        im[new_points_o[0],new_points_o[1]-1] = 230
        im[new_points_o[0]-1,new_points_o[1]-1] = 230
    return img, rew
