import logging
import os.path as osp
import time

import easy_tf_log
import numpy as np
from numpy.testing import assert_equal
import tensorflow as tf

from reinforcement_learning.utils import RunningStat, batch_iter


class RewardPredictorEnsemble:
    """
    An ensemble of reward predictors and associated helper functions.
    """

    def __init__(self,
                 cluster_job_name,
                 core_network,
                 lr=1e-4,
                 cluster_dict=None,
                 batchnorm=False,
                 dropout=0.0
                 ):
        graph, self.sess = self.init_sess(cluster_dict, cluster_job_name)
        # Why not just use soft device placement? With soft placement,
        # if we have a bug which prevents an operation being placed on the GPU
        # (e.g. we're using uint8s for operations that the GPU can't do),
        # then TensorFlow will be silent and just place the operation on a CPU.
        # Instead, we want to say: if there's a GPU present, definitely try and
        # put things on the GPU. If it fails, tell us!
        if tf.test.gpu_device_name():
            worker_device = "/job:{}/task:0/gpu:0".format(cluster_job_name)
        else:
            worker_device = "/job:{}/task:0".format(cluster_job_name)
        device_setter = tf.compat.v1.train.replica_device_setter(
            cluster=cluster_dict,
            ps_device="/job:ps/task:0",
            worker_device=worker_device)
        with graph.as_default():
            with tf.device(device_setter):
                self.rp = RewardPredictorNetwork(
                    core_network=core_network,
                    dropout=dropout,
                    batchnorm=batchnorm,
                    lr=lr)
            
            self.init_op = tf.compat.v1.global_variables_initializer()

            self.saver = tf.compat.v1.train.Saver(max_to_keep=1, save_relative_paths=True)
            self.saver.restore(self.sess, 'tmp/model.ckpt')
            print('Loaded weights for rew pred correctly')

        self.n_steps = 0
        self.r_norm = RunningStat(shape=1)


    @staticmethod
    def init_sess(cluster_dict, cluster_job_name):
        graph = tf.Graph()
        cluster = tf.train.ClusterSpec(cluster_dict)
        config = tf.compat.v1.ConfigProto(gpu_options={'allow_growth': True})
        server = tf.compat.v1.train.Server(cluster, job_name=cluster_job_name, config=config)
        sess = tf.compat.v1.Session(server.target, graph)
        return graph, sess

    def add_summary_ops(self):
        summary_ops = []

        for pred_n, rp in enumerate(self.rps):
            name = 'reward_predictor_accuracy_{}'.format(pred_n)
            op = tf.compat.v1.summary.scalar(name, rp.accuracy)
            summary_ops.append(op)
            name = 'reward_predictor_loss_{}'.format(pred_n)
            op = tf.compat.v1.summary.scalar(name, rp.loss)
            summary_ops.append(op)

        mean_accuracy = tf.reduce_mean([rp.accuracy for rp in self.rps])
        op = tf.compat.v1.summary.scalar('reward_predictor_accuracy_mean', mean_accuracy)
        summary_ops.append(op)

        mean_loss = tf.reduce_mean([rp.loss for rp in self.rps])
        op = tf.compat.v1.summary.scalar('reward_predictor_loss_mean', mean_loss)
        summary_ops.append(op)

        summaries = tf.compat.v1.summary.merge(summary_ops)

        return summaries

    def init_network(self, load_ckpt_dir=None):
        if load_ckpt_dir:
            ckpt_file = tf.train.latest_checkpoint(load_ckpt_dir)
            if ckpt_file is None:
                msg = "No reward predictor checkpoint found in '{}'".format(
                    load_ckpt_dir)
                raise FileNotFoundError(msg)
            self.saver.restore(self.sess, ckpt_file)
            print("Loaded reward predictor checkpoint from '{}'".format(ckpt_file))
        else:
            self.sess.run(self.init_op)

    def save(self):
        ckpt_name = self.saver.save(self.sess,
                                    self.checkpoint_file,
                                    self.n_steps)
        print("Saved reward predictor checkpoint to '{}'".format(ckpt_name))

    def raw_rewards(self, obs):
        """
        Return (unnormalized) reward for each frame of a single segment
        from each member of the ensemble.
        """
        assert_equal(obs.shape[1:], (256, 256))
        n_steps = obs.shape[0]
        feed_dict = {}
        feed_dict[self.rp.training] = False
        feed_dict[self.rp.s1] = [obs]
        # This will return nested lists of sizes n_preds x 1 x nsteps
        # (x 1 because of the batch size of 1)
        rs = self.sess.run(self.rp.r1, feed_dict)
        # rs = np.array(rs)
        # Get rid of the extra x 1 dimension
        # rs = rs[:, 0, :]
        rs = np.reshape(rs, (rs.shape[1],))
        # print(rs.shape)
        # print(rs)
        assert_equal(rs.shape, (n_steps,))
        return rs

    def reward(self, obs):
        """
        Return (normalized) reward for each frame of a single segment.

        (Normalization involves normalizing the rewards from each member of the
        ensemble separately, then averaging the resulting rewards across all
        ensemble members.)
        """
        assert_equal(obs.shape[1:], (256, 256))
        n_steps = obs.shape[0]

        # Get unnormalized rewards

        ensemble_rs = self.raw_rewards(obs)
        logging.debug("Unnormalized rewards:\n%s", ensemble_rs)

        # Normalize rewards

        # Note that we implement this here instead of in the network itself
        # because:
        # * It's simpler not to do it in TensorFlow
        # * Preference prediction doesn't need normalized rewards. Only
        #   rewards sent to the the RL algorithm need to be normalized.
        #   So we can save on computation.

        # Page 4:
        # "We normalized the rewards produced by r^ to have zero mean and
        #  constant standard deviation."
        # Page 15: (Atari)
        # "Since the reward predictor is ultimately used to compare two sums
        #  over timesteps, its scale is arbitrary, and we normalize it to have
        #  a standard deviation of 0.05"
        # Page 5:
        # "The estimate r^ is defined by independently normalizing each of
        #  these predictors..."

        # We want to keep track of running mean/stddev for each member of the
        # ensemble separately, so we have to be a little careful here.
        assert_equal(ensemble_rs.shape, (self.n_preds, n_steps))
        ensemble_rs = ensemble_rs.transpose()
        assert_equal(ensemble_rs.shape, (n_steps, self.n_preds))
        for ensemble_rs_step in ensemble_rs:
            self.r_norm.push(ensemble_rs_step)
        ensemble_rs -= self.r_norm.mean
        ensemble_rs /= (self.r_norm.std + 1e-12)
        ensemble_rs *= 0.05
        ensemble_rs = ensemble_rs.transpose()
        assert_equal(ensemble_rs.shape, (self.n_preds, n_steps))
        logging.debug("Reward mean/stddev:\n%s %s",
                      self.r_norm.mean,
                      self.r_norm.std)
        logging.debug("Normalized rewards:\n%s", ensemble_rs)

        # "...and then averaging the results."
        rs = np.mean(ensemble_rs, axis=0)
        assert_equal(rs.shape, (n_steps, ))
        logging.debug("After ensemble averaging:\n%s", rs)

        return rs

    def preferences(self, s1s, s2s):
        """
        Predict probability of human preferring one segment over another
        for each segment in the supplied batch of segment pairs.
        """
        feed_dict = {}
        for rp in self.rps:
            feed_dict[rp.s1] = s1s
            feed_dict[rp.s2] = s2s
            feed_dict[rp.training] = False
        preds = self.sess.run([rp.pred for rp in self.rps], feed_dict)
        return preds

    def train(self, prefs_train, prefs_val, val_interval):
        """
        Train all ensemble members for one epoch.
        """
        print("Training/testing with %d/%d preferences" % (len(prefs_train),
                                                           len(prefs_val)))

        start_steps = self.n_steps
        start_time = time.time()

        for _, batch in enumerate(batch_iter(prefs_train.prefs,
                                             batch_size=32,
                                             shuffle=True)):
            self.train_step(batch, prefs_train)
            self.n_steps += 1

            if self.n_steps and self.n_steps % val_interval == 0:
                self.val_step(prefs_val)

        end_time = time.time()
        end_steps = self.n_steps
        rate = (end_steps - start_steps) / (end_time - start_time)
        easy_tf_log.tflog('reward_predictor_training_steps_per_second',
                          rate)

    def train_step(self, batch, prefs_train):
        s1s = [prefs_train.segments[k1] for k1, k2, pref, in batch]
        s2s = [prefs_train.segments[k2] for k1, k2, pref, in batch]
        prefs = [pref for k1, k2, pref, in batch]
        feed_dict = {}
        for rp in self.rps:
            feed_dict[rp.s1] = s1s
            feed_dict[rp.s2] = s2s
            feed_dict[rp.pref] = prefs
            feed_dict[rp.training] = True
        ops = [self.summaries, [rp.train for rp in self.rps]]
        summaries, _ = self.sess.run(ops, feed_dict)
        self.train_writer.add_summary(summaries, self.n_steps)

    def val_step(self, prefs_val):
        val_batch_size = 32
        if len(prefs_val) <= val_batch_size:
            batch = prefs_val.prefs
        else:
            idxs = np.random.choice(len(prefs_val.prefs),
                                    val_batch_size,
                                    replace=False)
            batch = [prefs_val.prefs[i] for i in idxs]
        s1s = [prefs_val.segments[k1] for k1, k2, pref, in batch]
        s2s = [prefs_val.segments[k2] for k1, k2, pref, in batch]
        prefs = [pref for k1, k2, pref, in batch]
        feed_dict = {}
        for rp in self.rps:
            feed_dict[rp.s1] = s1s
            feed_dict[rp.s2] = s2s
            feed_dict[rp.pref] = prefs
            feed_dict[rp.training] = False
        summaries = self.sess.run(self.summaries, feed_dict)
        self.test_writer.add_summary(summaries, self.n_steps)


class RewardPredictorNetwork:
    """
    Predict the reward that a human would assign to each frame of
    the input trajectory, trained using the human's preferences between
    pairs of trajectories.

    Network inputs:
    - s1/s2     Trajectory pairs
    - pref      Preferences between each pair of trajectories
    Network outputs:
    - r1/r2     Reward predicted for each frame
    - rs1/rs2   Reward summed over all frames for each trajectory
    - pred      Predicted preference
    """

    def __init__(self, core_network, dropout, batchnorm, lr):
        training = tf.compat.v1.placeholder(tf.bool)
        # Each element of the batch is one trajectory segment.
        # (Dimensions are n segments x n frames per segment x ...)
        s1 = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 256, 256))
        s2 = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 256, 256))
        # For each trajectory segment, there is one human judgement.
        pref = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))

        # Concatenate trajectory segments so that the first dimension is just
        # frames
        # (necessary because of conv layer's requirements on input shape)
        s1_unrolled = tf.reshape(s1, [-1, 256, 256, 1])
        s2_unrolled = tf.reshape(s2, [-1, 256, 256, 1])

        # Predict rewards for each frame in the unrolled batch
        _r1 = core_network(
            s=s1_unrolled,
            dropout=dropout,
            batchnorm=batchnorm,
            reuse=False,
            training=training)
        _r2 = core_network(
            s=s2_unrolled,
            dropout=dropout,
            batchnorm=batchnorm,
            reuse=True,
            training=training)

        # Shape should be 'unrolled batch size'
        # where 'unrolled batch size' is 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(_r1, 1)
        c2 = tf.assert_rank(_r2, 1)
        with tf.control_dependencies([c1, c2]):
            # Re-roll to 'batch size' x 'n frames per segment'
            __r1 = tf.reshape(_r1, tf.shape(s1)[0:2])
            __r2 = tf.reshape(_r2, tf.shape(s2)[0:2])
        # Shape should be 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(__r1, 2)
        c2 = tf.assert_rank(__r2, 2)
        with tf.control_dependencies([c1, c2]):
            r1 = __r1
            r2 = __r2

        # Sum rewards over all frames in each segment
        _rs1 = tf.reduce_sum(r1, axis=1)
        _rs2 = tf.reduce_sum(r2, axis=1)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_rs1, 1)
        c2 = tf.assert_rank(_rs2, 1)
        with tf.control_dependencies([c1, c2]):
            rs1 = _rs1
            rs2 = _rs2

        # Predict preferences for each segment
        _rs = tf.stack([rs1, rs2], axis=1)
        # Shape should be 'batch size' x 2
        c1 = tf.assert_rank(_rs, 2)
        with tf.control_dependencies([c1]):
            rs = _rs
        _pred = tf.nn.softmax(rs)
        # Shape should be 'batch_size' x 2
        c1 = tf.assert_rank(_pred, 2)
        with tf.control_dependencies([c1]):
            pred = _pred

        preds_correct = tf.equal(tf.argmax(pref, 1), tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(preds_correct, tf.float32))

        _loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=pref,
                                                           logits=rs)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_loss, 1)
        with tf.control_dependencies([c1]):
            loss = tf.reduce_sum(_loss)

        # Make sure that batch normalization ops are updated
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # Inputs
        self.training = training
        self.s1 = s1
        self.s2 = s2
        self.pref = pref

        # Outputs
        self.r1 = r1
        self.r2 = r2
        self.rs1 = rs1
        self.rs2 = rs2
        self.pred = pred

        self.accuracy = accuracy
        self.loss = loss
        self.train = train