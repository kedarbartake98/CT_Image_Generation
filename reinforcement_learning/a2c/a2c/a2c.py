import logging
import os.path as osp
import queue
import time

import cloudpickle
import easy_tf_log
import numpy as np
from numpy.testing import assert_equal
import tensorflow as tf
from PIL import Image
# from a2c import logger
# from a2c.common import explained_variance, set_global_seeds
from reinforcement_learning.pref_db import Segment

    
class Model(object):
    def __init__(self,
                 policy,
                 ob_space,
                 ac_space,
                 nenvs, # 1
                 nsteps, # 5 / 9 ?
#                  nstack,  
#                  num_procs,
                 lr_scheduler,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 alpha=0.99,
                 epsilon=1e-5):
#         config = tf.ConfigProto(
#             allow_soft_placement=True,
#             intra_op_parallelism_threads=num_procs,
#             inter_op_parallelism_threads=num_procs)
#         config.gpu_options.allow_growth = True
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        nbatch = nenvs * nsteps   # = 5

        A = tf.compat.v1.placeholder(tf.int32, [nbatch])
        ADV = tf.compat.v1.placeholder(tf.float32, [nbatch])
        R = tf.compat.v1.placeholder(tf.float32, [nbatch])
        LR = tf.compat.v1.placeholder(tf.float32, [])

        step_model = policy(
            sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(
            sess, ob_space, ac_space, nenvs, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=train_model.pi, labels=A) # pi (nenvs, action_space.n) - action logits of policy ; not returned by policy
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R)) #vf = value ; not returned
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        def train(obs, rewards, actions, values):
            advs = rewards - values
            n_steps = len(obs)
            for _ in range(n_steps):
                cur_lr = lr_scheduler.value()
            td_map = {
                train_model.X: obs,
                A: actions,
                ADV: advs,
                R: rewards,
                LR: cur_lr
            }
#             if states != []:
#                 td_map[train_model.S] = states
# #                 td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train], td_map)
            return policy_loss, value_loss, policy_entropy, cur_lr

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
#         self.initial_state = step_model.initial_state
        self.sess = sess
        # Why var_list=params?
        # Otherwise we'll also save optimizer parameters,
        # which take up a /lot/ of space.
        # Why save_relative_paths=True?
        # So that the plain-text 'checkpoint' file written uses relative paths,
        # which seems to be needed in order to avoid confusing saver.restore()
        # when restoring from FloydHub runs.
        self.saver = tf.compat.v1.train.Saver(
            max_to_keep=1, var_list=params, save_relative_paths=True)
        tf.compat.v1.global_variables_initializer().run(session=sess)

    def load(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def save(self, ckpt_path, step_n):
        saved_path = self.saver.save(self.sess, ckpt_path, step_n)
        print("Saved policy checkpoint to '{}'".format(saved_path))


class Runner(object):  # Trains workers, does actions etc and pushed clips for prefs to frontend
    def __init__(self,
                 env,
                 model,
                 nsteps,
#                  nstack,
                 gamma,
                 gen_segments,
                 seg_pipe,
                 reward_predictor#,
#                  episode_vid_queue
                ):
        self.env = env
        self.model = model
#         nh, nw, nc = env.observation_space.shape
        nh = env.observation_space.shape[0]
        nenvs = env.nenvs # 8 
#         self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack) nstack  = 1, 
#         since we are not looking back 4 frames
        self.batch_ob_shape = (nenvs * nsteps, nh)
#         self.obs = np.zeros((nenv, nh), dtype=np.uint8)
#         # The first stack of 4 frames: the first 3 frames are zeros,
#         # with the last frame coming from env.reset().
        # self.obs, self.states = env.reset()
#         self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
#         self.states = model.initial_state
        # self.dones = [False for _ in range(nenv)]  # [0]

        self.gen_segments = gen_segments
        self.segment = Segment()
        self.seg_pipe = seg_pipe

        # self.orig_reward = [0 for _ in range(nenv)]  # [0]
        self.reward_predictor = reward_predictor

#         self.episode_frames = []
#         self.episode_vid_queue = episode_vid_queue

    def update_segment_buffer(self, mb_states, mb_rewards, mb_dones):

        # print('#'*100)
        # print("Called UPDATE SEGMENT Buffer")
        # Segments are only generated from the first worker.
        # Empirically, this seems to work fine.
        for e0_states, e0_rew, e0_dones in zip(mb_states, mb_rewards, mb_dones):
#             e0_obs = mb_obs[0]
#             e0_rew = mb_rewards[0]
#             e0_dones = mb_dones[0]
#             assert_equal(e0_obs.shape, (self.nsteps, 91))  
            assert_equal(e0_states.shape, (self.nsteps, 121)) # 50 + 50 + 1 + 20  
            assert_equal(e0_rew.shape, (self.nsteps, ))
            assert_equal(e0_dones.shape, (self.nsteps, ))

            for step in range(self.nsteps):
                # print("Called UPDATE SEGMENT Buffer 2")
                self.segment.append(np.copy(e0_states[step]), np.copy(e0_rew[step]))
                # print('Segment OBS')
                # # print(self.norm_obs_st)
                # print(self.segment.frames)
                if len(self.segment) == 25 or e0_dones[step]:
                    while len(self.segment) < 25:  # maybe 9 max?
                        # Pad to 25 steps long so that all segments in the batch
                        # have the same length.
                        # Note that the reward predictor needs the full frame
                        # stack, so we send all frames.
                        self.segment.append(e0_states[step], 0)
                    self.segment.finalise()
                    try:
                        # print("Called UPDATE SEGMENT Buffer 3")
                        self.seg_pipe.put(self.segment, block=False)
                        # print("Called UPDATE SEGMENT Buffer 3.5")

                    except queue.Full:
                        pass
                        # print("Called UPDATE SEGMENT Buffer 4")
                        ### ??? we should wait for half a second or break from the 
                        ### entire eight segments
                        # If the preference interface has a backlog of segments
                        # to deal with, don't stop training the agents. Just drop
                        # the segment and keep on going.
                    self.segment = Segment()

    def run(self):
        self.obs, self.states = self.env.reset()
        nenvs = self.env.nenvs
        self.dones = [False for _ in range(nenvs)] 
        mb_obs, mb_states, mb_rewards, mb_actions, mb_values, mb_dones = \
            [], [], [], [], [], []
#         mb_states = self.states

        # Run for nsteps steps in the environment
        for _ in range(self.nsteps):
            actions, values, _ = self.model.step(self.obs)
            mb_obs.append(np.copy(self.obs))
            mb_states.append(np.copy(self.states))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            # len({obs, rewards, dones}) == nenvs
            obs, rewards, dones, states = self.env.step(actions) # remove rewards
#             self.states = states
            self.dones = dones
#             for n, done in enumerate(dones):
#                 if done:
#                     self.obs[n] = self.obs[n] * 0
            # SubprocVecEnv automatically resets when done  ??? -- Resolved
            self.obs = obs
            self.states = states
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        # i.e. from nsteps, nenvs to nenvs, nsteps
        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0)
        mb_states = np.asarray(mb_states, dtype=np.float32).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
#         mb_masks = mb_dones[:, :-1]
        # The first entry was just the init state of 'dones' (all False),
        # before we'd actually run any steps, so drop it.
        mb_dones = mb_dones[:, 1:]

        if self.gen_segments:
            self.update_segment_buffer(mb_states, mb_rewards, mb_dones)

        assert_equal(mb_states.shape, (nenvs, self.nsteps, 121)) # this is where the shap.....
        mb_states_allenvs = mb_states.reshape(nenvs * self.nsteps, 121)
        mb_images_allenvs, rew = vectors_to_images(mb_states_allenvs)
        if mb_images_allenvs.shape[0] == 2:
            return np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2)), np.zeros((2,2))
        # with open('test.npy', 'wb') as f:
        #     np.save(f, mb_images_allenvs)
        # f.close()
        # if mb_images_allenvs == None:
        #     rewards_allenvs = np.zeros((nenvs * self.nsteps,))
        #     print('NOOONE')
        # else:
        assert_equal(mb_images_allenvs.shape, (nenvs * self.nsteps, 256, 256))
        rewards_allenvs = self.reward_predictor.raw_rewards(mb_images_allenvs)
        rewards_allenvs = np.where(rew == 0, 0.0, rewards_allenvs)
        assert_equal(rewards_allenvs.shape, (nenvs * self.nsteps, ))
        # print('NOOOOT    NOOONE')
        ### TODO : make -1/-2.. and 1 # Can also use np.where
#             mb_rewards = np.multiply(mb_rewards*(rewards_allenvs.reshape(nenvs, self.nsteps)))
        rewards_allenvs = rewards_allenvs.reshape(nenvs, self.nsteps)
        mb_rewards = np.where(mb_rewards==-1, 0.0, rewards_allenvs)
        assert_equal(mb_rewards.shape, (nenvs, self.nsteps))

        logging.debug("Predicted rewards:\n%s", mb_rewards)

        # Save frames for episode rendering
#         if self.episode_vid_queue is not None:
#             self.update_episode_frame_buffer(mb_obs, mb_dones)

        # Discount rewards
        mb_obs = mb_obs.reshape(self.batch_ob_shape)
        last_values = self.model.value(self.obs).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(
                zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                # Make sure that the first iteration of the loop inside
                # discount_with_dones picks up 'value' as the initial       #???
                # value of r
                rewards = discount_with_dones(rewards + [value],
                                              dones + [0],
                                              self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
#         mb_masks = mb_masks.flatten()
        return mb_obs, mb_rewards, mb_actions, mb_values, mb_states


def learn(policy,
          env,
#           seed,  ### ??? -- resolved
          start_policy_training_pipe,  
          ckpt_save_dir,
          lr_scheduler,
          nsteps=9,
          n_envs=8,
          total_timesteps=int(80e6),
          vf_coef=0.5,
          ent_coef=0.01,
          max_grad_norm=0.5,
          epsilon=1e-5,
          alpha=0.99,
          gamma=0.99,
          log_interval=100,
          ckpt_save_interval=1000,
          ckpt_load_dir=None,
          gen_segments=False,
          seg_pipe=None,
          reward_predictor=None):
#           episode_vid_queue=None):

    tf.compat.v1.reset_default_graph()
#     set_global_seeds(seed)

    nenvs = env.nenvs
    ob_space = env.observation_space
    ac_space = env.action_space
    # num_procs = len(env.remotes)  # HACK
# 
    def make_model():
        return Model(
            policy=policy,
            ob_space=ob_space,
            ac_space=ac_space,
            nenvs=nenvs,
            nsteps=nsteps,
#             nstack=nstack,
            # num_procs=num_procs,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            lr_scheduler=lr_scheduler,
            alpha=alpha,
            epsilon=epsilon)

    def explained_variance(ypred,y):
        """
        Computes fraction of variance that ypred explains about y.
        Returns 1 - Var[y-ypred] / Var[y]
        interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero
        """
        assert y.ndim == 1 and ypred.ndim == 1
        vary = np.var(y)
        return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

    def store_images(mb_states, update):
        vector_img_1 = mb_states[0][8][:50]
        vector_img_2 = mb_states[0][8][50:100]
        vector_img_middle = [] 
        int_lvl = mb_states[0][8][100]
        if int_lvl == 0:
            vector_img_middle = 0.7*vector_img_1 + 0.3*vector_img_2  # for slider at 0.3
        elif int_lvl == 1:
            vector_img_middle = 0.5*vector_img_1 + 0.5*vector_img_2
        else:
            vector_img_middle = 0.3*vector_img_1 + 0.7*vector_img_2

        img1 = vector_to_image(vector_img_1)
        img2 = vector_to_image(vector_img_2)
        img_mid = vector_to_image(vector_img_middle)
        # eight_images = [] 
        im = Image.fromarray(np.uint8(img1))
        im.save(str(update) + '_1.png')
        im = Image.fromarray(np.uint8(img2))
        im.save(str(update) + '_2.png')
        im = Image.fromarray(np.uint8(img_mid))
        im.save(str(update) + '_mid.png')

        for i,e0_states in enumerate(mb_states):
            assert_equal(e0_states.shape, (9, 121)) # nsteps, 121 
            torso = e0_states[8][101:]
            torso = np.append(torso, vector_img_middle[20:])
            # eight_images.append(vector_to_image(torso))
            im = Image.fromarray(np.uint8(vector_to_image(torso)))
            im.save(str(update) + '_res_' + str(i+1) + '.png')

    print("Initialising policy...")
    if ckpt_load_dir is None:
        model = make_model()
    else:
        with open(osp.join(ckpt_load_dir, 'make_model.pkl'), 'rb') as fh:
            make_model = cloudpickle.loads(fh.read())
        model = make_model()

        ckpt_load_path = tf.train.latest_checkpoint(ckpt_load_dir)
        model.load(ckpt_load_path)
        print("Loaded policy from checkpoint '{}'".format(ckpt_load_path))

    ckpt_save_path = osp.join(ckpt_save_dir, 'policy.ckpt')

    runner = Runner(env=env,
                    model=model,
                    nsteps=nsteps,
#                     nstack=nstack,
                    gamma=gamma,
                    gen_segments=gen_segments,
                    seg_pipe=seg_pipe,
                    reward_predictor=reward_predictor#,
#                     episode_vid_queue=episode_vid_queue
                   )

    # nsteps: e.g. 5
    # nenvs: e.g. 16
    nbatch = nenvs * nsteps
    fps_tstart = time.time()
    fps_nsteps = 0

    print("Starting workers")

    # Before we're told to start training the policy itself,
    # just generate segments for the reward predictor to be trained with    #not needed since pretrained
    while True:
        runner.run()
        try:
            start_policy_training_pipe.get(block=False)  ### ???
        except queue.Empty:
            continue
        else:
            break

    print("Starting policy training")

    tot_plc_loss = 0
    tot_value_loss = 0
    tot_ev_ne = 0
    tot_ev_po = 0
    c = 0
    c_ev_po = 0
    c_ev_ne = 0
    for update in range(1, total_timesteps // nbatch + 1):
        # Run for nsteps
        obs, rewards, actions, values, mb_states = runner.run()
        if obs.shape[0] == 2:
            continue
        # print(rewards)
        # break

        policy_loss, value_loss, policy_entropy, cur_lr = model.train(
            obs, rewards, actions, values)
        ev = explained_variance(values, rewards)  ### ???
        tot_plc_loss += policy_loss
        tot_value_loss += value_loss
        if ev > 0:
            tot_ev_po += ev
            c_ev_po += 1
        if ev < 0:
            tot_ev_ne += ev
            c_ev_ne += 1
        c += 1

        fps_nsteps += nbatch

        if update % log_interval == 0 and update != 0:
            fps = fps_nsteps / (time.time() - fps_tstart)
            fps_nsteps = 0
            fps_tstart = time.time()

            print("Trained policy for {} time steps".format(update * nbatch))
#             store_images(mb_states, update)

            print(tot_plc_loss / c, tot_value_loss / c, tot_ev_po / c_ev_po, tot_ev_ne / c_ev_ne)
            tot_plc_loss = 0
            tot_value_loss = 0
            tot_ev_po = 0
            tot_ev_ne = 0
            c = 0
            c_ev_ne = 0
            c_ev_po = 0
            # logger.record_tabular("nupdates", update)

        if update != 0 and update % ckpt_save_interval == 0:
            model.save(ckpt_save_path, update)

    model.save(ckpt_save_path, update)


from reinforcement_learning.a2c.a2c.utils import (cat_entropy, discount_with_dones,
                           find_trainable_variables, mse, vectors_to_images, vector_to_image)
