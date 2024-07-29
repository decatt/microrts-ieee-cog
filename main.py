import numpy as np
from nets import ActorCriticMix, ActorCritic
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from agent import MixAgent
import argparse
import torch

ai_dict = {
    "coacAI": microrts_ai.coacAI,
    "rojo": microrts_ai.rojo,
    "randomAI": microrts_ai.randomAI,
    "passiveAI": microrts_ai.passiveAI,
    "workerRushAI": microrts_ai.workerRushAI,
    "lightRushAI": microrts_ai.lightRushAI,
}

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--map_path', type=str, default='maps/16x16/TwoBasesBarracks16x16.xml')
parser.add_argument('--op_ai', type=str, default='coacAI')
args = parser.parse_args()

op_ai = ai_dict[args.op_ai]

if __name__=="__main__":
    map_path = args.map_path
    num_envs = 1
    net = ActorCriticMix()
    adp = False
    adp_net = None
    action_space = [256, 6, 4, 4, 4, 4, 7, 49]
    
    if map_path == "maps/16x16/basesWorkers16x16A.xml":
        net.load_state_dict(torch.load("expert_model\\tbb16\ppo_model_mix_TwoBasesBarracks16x16909_7500.pkl"))
        w = 16
        h = 16
        weight = 2
        action_space = [w*h, 6, 4, 4, 4, 4, 7, 49]
        adp_net = ActorCritic(32*6*6, action_space)
        adp_net.load_state_dict(torch.load("expert_model\\tbb16\\adp_mix_rl_basesWorkers16x16A_2514752095000.pkl"))
        adp = True
    if map_path == "maps/8x8/FourBasesWorkers8x8.xml":
        net.load_state_dict(torch.load("expert_model\\16a\ppo_model_mix_522_2500.pt"))
        w = 8
        h = 8
        weight = 2
        action_space = [w*h, 6, 4, 4, 4, 4, 7, 49]
        adp_net = ActorCritic(32*2*2, action_space)
        adp_net.load_state_dict(torch.load("expert_model\\16a\\adp_mix_rl_FourBasesWorkers8x8_265690543.pkl"))
        adp = True
    if map_path == "maps/16x16/basesWorkers16x16noResources.xml":
        net.load_state_dict(torch.load("expert_model\\16b\ppo_model_mix_522_2500.pt"))
        w = 16
        h = 16
        weight = 2
        action_space = [w*h, 6, 4, 4, 4, 4, 7, 49]
        adp_net = ActorCritic(32*6*6, action_space)
        adp_net.load_state_dict(torch.load("expert_model\\16a\\adp_mix_rl_basesWorkers16x16noResources_2_56629507.pkl"))
        adp = True
    if map_path == "maps/16x16/TwoBasesBarracks16x16.xml":
        net.load_state_dict(torch.load("expert_model\\tbb16\ppo_model_mix_TwoBasesBarracks16x16909_7500.pkl"))
        w = 16
        h = 16
        weight = 2
        action_space = [w*h, 6, 4, 4, 4, 4, 7, 49]
        adp_net = ActorCritic(32*6*6, action_space)
        adp_net.load_state_dict(torch.load("expert_model\\tbb16\\adp_mix_rl_TwoBasesBarracks16x16_2574848556000.pkl"))
        adp = True
    else:
        net.load_state_dict(torch.load("expert_model\ppo_model_mix_522_2500.pt"))
        adp=False
    


    agent = MixAgent(net, adp_net, action_space)
    env = MicroRTSVecEnv(
                num_envs=num_envs,
                max_steps=5000,
                ai2s=[op_ai for _ in range(num_envs)],
                map_path=map_path,
                reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
            )
    obs = env.reset()
    for _ in range(10000):
        env.render()

        if adp == False:
            actions = agent.get_action(obs, env)
        else:
            actions = agent.get_adp_action(obs, env, weight)
        obs, reward, done, info = env.step(actions)