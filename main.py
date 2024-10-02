import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

from Environment import MEC_Env
from Model.MADDPG.maddpg import MADDPG
from Model.MADDPG.buffer import MultiAgentReplayBuffer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ue", type=int, default=20, help="num of user equipments")
    parser.add_argument("--num_es", type=int, default=4, help="num of edge servers")
    parser.add_argument("--num_episodes", type=int, default=2000, help="num of episodes")
    parser.add_argument("--max_steps", type=int, default=100, help="num of steps in an episode")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initial environment
    mec_env = MEC_Env.init_env(side_len=100,
                               station_num=args.num_es,
                               user_num=args.num_ue,
                               station_cpu_frequency=20,
                               station_band_width=10,
                               station_noise_power=6 * pow(10, -8),
                               user_cpu_frequency=1,
                               user_trans_power=1)
                                                     
    # initial actor-critic agents
    n_agents = mec_env.get_agent_num()  # 4
    actor_dims = [mec_env.get_obs_dim] * n_agents  # [35, 35, 35, 35]
    critic_dims = sum(actor_dims)  # 140
    n_actions = mec_env.get_action_dim()  # 10

    maddpg_agents = MADDPG(actor_dims=actor_dims, 
                           critic_dims=critic_dims, 
                           n_agents=n_agents,
                           n_actions=n_actions,
                           alpha=0.01,
                           beta=0.01,
                           fc1=64,
                           fc2=64,
                           gamma=0.99,
                           tau=0.01,
                           chkpt_dir='weights/maddpg/')
    
    # initial replay buffer
    memory = MultiAgentReplayBuffer(max_size=1000000,
                                    critic_dims=critic_dims,
                                    actor_dims=actor_dims,
                                    n_actions=n_actions,
                                    n_agents=n_actions,
                                    batch_size=1024)

    if args.evaluate:
        maddpg_agents.load_checkpoint()

    total_step = 0
    for i in tqdm(range(args.num_episodes)):
        print("episode_index = {i}")
        
        episode_step = 0
        episode_energys = []
        episode_delay = []
        episode_success_ratio = []
        score = 0
        done = [False] * n_agents  # whether agent finished

        mec_env.reset_env()

        while not any(done):
            if episode_step >= args.max_steps:  # 100
                done = [True] * n_agents

            obs = mec_env.get_obs()
            actions = maddpg_agents.choose_action(obs)
            actions_pos = maddpg_agents.pos_process_action(actions)
            obs_, step_reward, step_energy, step_delay, step_workload, step_successs = \
                  mec_env.step(now_slot=episode_step, slot_size=1, actions_pos=actions_pos)
            state  = obs.flatten()
            state_ = obs_.flatten()
            memory.store_transition(raw_obs=obs, state=state, action=actions, reward=step_reward,
                                    raw_obs_=obs_, state_=state_, done=done)

            if total_step % 1000 == 0 and not args.evaluate:
                maddpg_agents.learn(memory)

            obs = obs_
            score += step_reward
            total_step += 1
            episode_step += 1
            episode_energys.append(step_energy)
            episode_delay.append(step_delay)
            episode_success_ratio.append(step_success_ratio)

        eposide_avg_energys = sum(episode_energys) / len(episode_energys)
        eposide_avg_delay = sum(episode_delay) / len(episode_delay)
        eposide_avg_success_ratio = sum(episode_success_ratio) / len(episode_success_ratio)




