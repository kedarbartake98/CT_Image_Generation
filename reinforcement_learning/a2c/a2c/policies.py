import numpy as np
import tensorflow as tf
from reinforcement_learning.a2c.a2c.utils import fc, sample

class MlpPolicy(object):

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 nenv,
                 nsteps,
#                  nstack,
                 reuse=False):
        nbatch = nenv*nsteps
        nh = ob_space.shape
        ob_shape = (nbatch, nh)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            x = tf.cast(X, tf.float32)

            # Only look at the most recent frame
#             x = x[:, :, :, -1]

#             h = x.get_shape()[1:]
#             x = tf.reshape(x, [-1, int(w * h)])
            x = fc(x, 'fc1', nh=256, init_scale=np.sqrt(2))
            x = fc(x, 'fc2', nh=128, init_scale=np.sqrt(2))
            x = fc(x, 'fc3', nh=64,  init_scale=np.sqrt(2))
            pi = fc(x, 'pi', nact, act=lambda x: x)
            vf = fc(x, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
#         a0, a1 = sample(pi)
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step(ob):
#             act1, act2, v = sess.run([a0, a1, v0], {X: ob})
            act1, v = sess.run([a0, v0], {X: ob})
#             return act1, act2, v, []  # dummy state
            return act1, v, []  # dummy state

        def value(ob):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value