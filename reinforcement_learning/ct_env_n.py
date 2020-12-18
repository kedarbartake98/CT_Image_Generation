import argparse
import gym
from gym import spaces
from numpy.testing import assert_equal
from reinforcement_learning.utils import get_normalized_organ_data, \
     load_componets, load_samples, get_first_grp_struct, \
     get_second_grp_struct, get_third_grp_struct

import numpy as np

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, nenvs):#, device):
        super(CustomEnv, self).__init__()
        # self.loadData()
        self.nenvs = nenvs
        #TODO: Normalization needed -- DONE
#         self.curves # [
#                         [ #image0
#                          [[x,y],....],[#organ1.........],.....
#                         ],......#100,000
#                        ]
        self.samples = load_samples()
        self.norm_samples, org_ests = get_normalized_organ_data(self.samples)
        self.torso_est = org_ests[0]
#         self.samples # [
#                         [ #image0
#                          [1.54, 1.2,....],[#organ1, 1.54, 1.2,....],.....#6
#                         ],......#100,000
#                        ]
        self.max_list = self.torso_est.inverse_transform(np.ones((1,20)))
        self.min_list = self.torso_est.inverse_transform(np.zeros((1,20)))
        self.estimators = load_componets()
        self.co_in = get_first_grp_struct()
        self.co_in123 = get_second_grp_struct()
        self.co_in45 = get_third_grp_struct()

        self.current_step = 0
        self.MAX_STEPS = 9
#         self.device = device

        
#         self.all_spaces = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(8)))
        self.all_spaces = spaces.Discrete(65)
#         self.action_space = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(8)))
        self.action_space = spaces.Discrete(65)
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1.0, 
                                                          shape=(self.torso_d*3+self.lung_d*3+self.sp_d*2+1,), 
                                                          dtype=np.float16) ))

    def step(self, action):
        # Execute one time step within the environment
        assert_equal(action.shape, (self.nenvs, 2))
        self._take_action(action)
        self.current_step += 1
        if self.current_step > self.MAX_STEPS:  #RESET
            # we are not checking for 'no action' action since longer routes 
            # would mean larger rewards
            self.current_step = 0
            done = [True for _ in range(self.nenvs)]
        else:
            done = [False for _ in range(self.nenvs)]
        # comment since we are handling it later
#         delay_modifier = (self.current_step / self.MAX_STEPS) #DEFINE
        reward = np.zeros(self.nenvs)
        curves_torso = np.reshape((self.curves_torsos*255.5 + 255.5), (self.nenvs, -1, 2)).astype(int).tolist()
        for env_n in range(self.nenvs):
            torso = Polygon([curves_torso[env_n][index] for index in self.co_in])

            if torso.contains(self.leftlung) and torso.contains(self.rightlung) and torso.contains(self.heart) \
            and torso.contains(self.esophagus):
                reward[env_n] = 0
            else:
                reward[env_n] = -1
        
        self.norm_obs_st = np.append(self.torso_est.transform(self.torsos), self.const_arr, axis=1)
        self.state = np.append(self.input_imgs, self.torsos, axis=1)        
#         reward = reward * delay_modifier
        return self.norm_obs_st, reward, done, self.state
    
    def reset(self):
        self.current_step = 0
        self.selected_idx1 = np.random.randrange(len(self.samples))
        self.selected_idx2 = np.random.randrange(len(self.samples))
        self.int_level = np.random.randrange(3)
        self.samples_img1 = self.samples[self.selected_idx1]
        self.samples_img2 = self.samples[self.selected_idx2]
        self.samples_img1_chain = [ev for org in self.samples_img1 for ev in org]
        self.samples_img2_chain = [ev for org in self.samples_img2 for ev in org]
        self.input_imgs = np.asarray(samples_img1_chain.extend(samples_img2_chain).append(int_level))
        self.input_imgs = np.expand_dims(self.norm_const_arr, axis=0).repeat(self.nenvs, 0)
        self.norm_samples_img1 = self.norm_samples[self.selected_idx1]
        self.norm_samples_img2 = self.norm_samples[self.selected_idx2]
#         self.curves_img1 = self.curves[self.selected_idx1]
#         self.curves_img2 = self.curves[self.selected_idx2]
        self.samples_int = [] 
        self.norm_samples_int = [] 
        for sample1,sample2 in zip(self.samples_img1[1:], self.samples_img2[1:]):
            if int_level == 0:
                self.samples_int.append(0.7*sample1 + 0.3*sample2)
            elif int_level == 1:
                self.samples_int.append(0.5*sample1 + 0.5*sample2)
            else:
                self.samples_int.append(0.3*sample1 + 0.7*sample2)
        
        for sample1,sample2 in zip(self.norm_samples_img1[1:], self.norm_samples_img2[1:]):
            if int_level == 0:
                self.norm_samples_int.append(0.7*sample1 + 0.3*sample2)
            elif int_level == 1:
                self.norm_samples_int.append(0.5*sample1 + 0.5*sample2)
            else:
                self.norm_samples_int.append(0.3*sample1 + 0.7*sample2)
        self.norm_samples_chain = [ev for org in self.norm_samples_int for ev in org]
        if int_level == 0:
            self.norm_torsos = np.asarray(0.7*self.norm_samples_img1[0] + 0.3*self.norm_samples_img2[0])
        elif int_level == 1:
            self.norm_torsos = np.asarray(0.5*self.norm_samples_img1[0] + 0.5*self.norm_samples_img2[0])
        else:
            self.norm_torsos = np.asarray(0.3*self.norm_samples_img1[0] + 0.7*self.norm_samples_img2[0])
        self.norm_torsos = np.expand_dims(self.torsos, axis=0)
        self.norm_torsos = np.repeat(self.torsos, self.nenvs, 0)
        self.torsos = self.torso_est.inverse_transform(self.norm_torsos) 
        
        self.curves_int = []
        for org in range(5):        
            curves_o = self.estimators[org+1].mean_
            for i,val in enumerate(self.samples_int[org]):
                curves_o = curves_o + self.estimators[org+1].components_[i]*val
            curves_o = np.reshape((curves_o*255.5 + 255.5), (-1, 2)).astype(int).tolist()
            self.curves_int.append(curves_o)
            
        self.leftlung = Polygon([self.curves_int[0][index] for index in self.co_in123]) 
        self.rightlung = Polygon([self.curves_int[1][index] for index in self.co_in123])
        self.heart = Polygon([self.curves_int[2][index] for index in self.co_in123])
        self.spinalcord = Polygon([self.curves_int[3][index] for index in self.co_in45])
        self.esophagus = Polygon([self.curves_int[4][index] for index in self.co_in45])
        
        assert_equal(self.torsos, (self.nenvs, 20))
        norm_tor1 = self.norm_samples_img1[0]
        norm_tor2 = self.norm_samples_img2[0]
        self.norm_const_arr = np.asarray(norm_samples_chain.extend(norm_tor1).extend(norm_tor2).append(int_level))
        self.norm_const_arr = np.expand_dims(self.norm_const_arr, axis=0).repeat(self.nenvs, 0)
        self.norm_obs_st = np.append(self.norm_torsos, self.norm_const_arr, axis=1)
        self.state = np.append(self.input_imgs, self.torsos, axis=1)
        assert_equal(self.norm_obs_st, (self.nenvs, 91))
        
        self.curves_torsos = np.matmul(np.asarray(self.estimators[0].components_),
                                      self.torsos) + \
                                    np.asarray(self.estimators[0].mean_).repeat(self.nenvs, 0)
        assert_equal(curves_torsos, (self.nenvs, 72)) # len self.estimators[0].components_[0]

        return self.norm_obs_st, self.state

    def _take_action(self, action):
        
        assert_equal(action, (self.nenvs, 1))
        eig_cmp = action[:,0] / 8  #[0....7,..,8]
        eig_mod = action[:,1] % 8  #[0....7,....]
        assert_equal(eig_cmp, (self.nenvs, ))
        assert_equal(eig_mod, (self.nenvs, ))
#         if eig_cmp != 8:
        tor1_ev = np.asarray(self.samples_img1)[0][eig_cmp]
        tor2_ev = np.asarray(self.samples_img2)[0][eig_cmp]
        tor3_ev = self.torsos[range(self.nenvs),eig_cmp]
        assert_equal(tor1_ev, (self.nenvs, ))
        assert_equal(tor2_ev, (self.nenvs, ))
        assert_equal(tor3_ev, (self.nenvs, ))
        max_ev = max_list[eig_cmp]
        min_ev = min_list[eig_cmp]
        assert_equal(max_ev, (self.nenvs, ))
        assert_equal(min_ev, (self.nenvs, ))
        eig_mod_01 = eig_mod % 2
        em = np.where(eig_mod_01==0, 0.1, 0.3)
        em = np.where(eig_cmp==8, 0, em)
        emv = np.where(int(eig_mod / 2) % 4==0, min_ev, max_ev)
        emv = np.where(int(eig_mod / 2) % 4==1, tor1_ev, emv)
        emv = np.where(int(eig_mod / 2) % 4==2, tor2_ev, emv)
        new_ev = emv*em + tor3_ev*(1 - em)
#             modif = [min_ev, tor1_ev, tor2_ev, max_ev]
#             if eig_mod % 2 == 0:
#                 new_ev = modif[int(eig_mod / 2) % 4]*0.1 + tor3_ev*0.9 # ??? other way around?
#             else:
#                 new_ev = modif[int(eig_mod / 2) % 4]*0.3 + tor3_ev*0.7 # can make diff settings here
#             self.samples_int[0][eig_cmp] = new_ev
#             self.samples_chain[eig_cmp] = new_ev
        self.torsos[range(self.nenvs),eig_cmp] = new_ev
    
        self.curves_torsos = np.matmul(np.asarray(self.estimators[0].components_),
                                  self.torsos) + \
                                np.asarray(self.estimators[0].mean_).repeat(self.nenvs, 0)
        assert_equal(curves_torsos, (self.nenvs, 72)) # len self.estimators[0].components_[0]
        # ??? maybe made faster by add/sub
