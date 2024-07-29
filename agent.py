import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from collections import deque
import numpy as np
import torch.nn.functional as F
from state_data_process import process_states, extract_centered_regions
from nets import ActorCriticMix, ActorCritic
from utils import MaskedCategorical, remake_mask


class MixAgent:
    def __init__(self,net:ActorCriticMix, adp_net:ActorCritic=None, action_space=[256,6, 4, 4, 4, 4, 7, 49]) -> None:
        self.net = net
        self.adp_net = adp_net
        self.action_space = action_space
   
    """
    obs: state of gym-microrts, shape = (num_env, height, width, feature_size=27)
    units: an available unit for each env, shape = (num_env, 1)
    action_mask_list: available action for each unit shape = (num_env, action_space)
    """
    @torch.no_grad()
    def get_action(self,obs,env):
        num_envs = obs.shape[0]

        unit_masks = np.array(env.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)
        unit_list = []
        for unit_mask in unit_masks:
            if np.sum(unit_mask) == 0:
                unit_list.append(0)
            else:
                unit_list.append(np.random.choice(np.where(unit_mask == 1)[0]))
        action_mask_list = np.array(env.vec_client.getUnitActionMasks(np.array(unit_list))).reshape(num_envs, -1)
        units =np.array(unit_list)

        action_mask_list = remake_mask(action_mask_list)
        cnn_states = extract_centered_regions(obs, units)
        linears_states = process_states(obs, units)

        distris,_ = self.net(cnn_states,linears_states)
        distirs = torch.split(distris,[6, 4, 4, 4, 4, 7, 49], dim=1)
        distris = [MaskedCategorical(distir) for distir in distirs]
        
        action_components = [torch.Tensor(units)]

        action_masks = torch.split(torch.Tensor(action_mask_list), [6, 4, 4, 4, 4, 7, 49], dim=1)
        action_components +=  [dist.update_masks(action_mask).sample() for dist, action_mask in zip(distris,action_masks)]
            
        actions = torch.stack(action_components)
        
        return actions.T.cpu().numpy().astype(int)
    
    
    @torch.no_grad()
    def get_adp_action(self,obs,env,weight=2):
        num_envs = obs.shape[0]
        base_dist = self.get_onehot_distribution(obs,env)
        states = torch.Tensor(obs)
        distris = self.adp_net.get_distris(states)

        base_dist = base_dist*weight

        distris = torch.tensor(base_dist, dtype=torch.float32) + distris

        distris = distris.split(self.action_space, dim=1)
        distris = [MaskedCategorical(dist) for dist in distris]
        
        unit_masks = np.array(env.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)
        unit_masks = torch.Tensor(unit_masks)
        distris[0].update_masks(unit_masks)
        
        units = distris[0].sample()
        action_components = [units]

        action_mask_list = np.array(env.vec_client.getUnitActionMasks(np.array(units))).reshape(num_envs, -1)
        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1) 
        
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris[1:],action_masks)]
            
        actions = torch.stack(action_components)
        
        return actions.T.cpu().numpy()
    
    @torch.no_grad()
    def get_distribution(self,obs,units,action_mask_list):
        action_mask_list = remake_mask(action_mask_list)
        cnn_states = extract_centered_regions(obs, units)
        linears_states = process_states(obs, units)
        distris,_ = self.net(cnn_states,linears_states)
        return distris
    
    @torch.no_grad()
    def get_base_action(self,obs,units,action_mask_list):
        action_mask_list = remake_mask(action_mask_list)
        cnn_states = extract_centered_regions(obs, units)
        linears_states = process_states(obs, units)

        distris,_ = self.net(cnn_states,linears_states)
        distirs = torch.split(distris,self.action_space[1:],dim=1)
        distris = [MaskedCategorical(distir) for distir in distirs]
        
        action_components = [torch.Tensor(units)]

        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1)
        action_components +=  [dist.update_masks(action_mask).sample() for dist, action_mask in zip(distris,action_masks)]
            
        actions = torch.stack(action_components)
        
        return actions.T.cpu().numpy().astype(int)
    
    @torch.no_grad()
    def get_onehot_distribution(self,obs,env):
        action_space_ = [obs.shape[1]*obs.shape[2], 6, 4, 4, 4, 4, 7, 49]
        num_envs = obs.shape[0]
        unit_masks = np.array(env.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)
        unit_list = []
        for unit_mask in unit_masks:
            if np.sum(unit_mask) == 0:
                unit_list.append(0)
            else:
                unit_list.append(np.random.choice(np.where(unit_mask == 1)[0]))
        action_mask_list = np.array(env.vec_client.getUnitActionMasks(np.array(unit_list))).reshape(num_envs, -1)

        actions = self.get_base_action(obs, np.array(unit_list), action_mask_list)
        res = np.zeros((num_envs, sum(action_space_)))
        for i in range(num_envs):
            res[i][actions[i][0]] = 1
            res[i][actions[i][1]+obs.shape[1]*obs.shape[2]] = 1
            res[i][actions[i][2]+obs.shape[1]*obs.shape[2]+6] = 1
            res[i][actions[i][3]+obs.shape[1]*obs.shape[2]+6+4] = 1
            res[i][actions[i][4]+obs.shape[1]*obs.shape[2]+6+4+4] = 1
            res[i][actions[i][5]+obs.shape[1]*obs.shape[2]+6+4+4+4] = 1
            res[i][actions[i][6]+obs.shape[1]*obs.shape[2]+6+4+4+4+4] = 1
            res[i][actions[i][6]+obs.shape[1]*obs.shape[2]+6+4+4+4+4+7] = 1
        return res