import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

# from Environment.env import MEC_Env
from Environment.env import MEC_Env
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
    mec_env = MEC_Env(side_len=100,
                      station_num=args.num_es,
                      user_num=args.num_ue,
                      station_cpu_frequency=20,
                      station_band_width=10,
                      station_noise_power=6 * pow(10, -8),
                      user_cpu_frequency=1,
                      user_trans_power=1)
    mec_env.init_env()
    mec_env.print()
    
                                                     
    # initial actor-critic agents
    n_agents    = mec_env.get_agent_num()  # 4
    actor_dims  = mec_env.get_obs_dim()  # 35
    critic_dims = actor_dims * n_agents  # 140
    n_actions   = mec_env.get_action_dim()  # 10
    
    maddpg_agents = MADDPG(n_agents=n_agents,
                           actor_dims=actor_dims, 
                           critic_dims=critic_dims, 
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
                                    n_agents=n_agents,
                                    actor_dims=actor_dims,
                                    critic_dims=critic_dims,
                                    n_actions=n_actions,
                                    batch_size=1024)

    if args.evaluate:
        maddpg_agents.load_checkpoint()

    total_step = 0
    print_interval = 500
    score_history = []
    score_best = 0
    for i in tqdm(range(args.num_episodes)):
        print(f"episode_index = {i}")
        
        score = 0
        episode_step = 0
        episode_energy = []
        episode_delay = []
        episode_workload = []
        episode_success_ratio = []
        done = [False] * n_agents  # whether agent finished

        mec_env.reset_env()
        obs = mec_env.get_obs()
        mec_env.print()

        while not any(done):
            if episode_step >= args.max_steps:  # 100
                done = [True] * n_agents

            actions = maddpg_agents.choose_action(obs)
            actions_pos = maddpg_agents.pos_process_action(actions)
            obs_, step_reward, step_energy, step_delay, step_workload, step_success_ratio = \
                  mec_env.step(now_slot=episode_step+1, slot_size=1, actions_pos=actions_pos)
            state  = obs.flatten()
            state_ = obs_.flatten()
            reward = step_reward.sum(axis=1)
            memory.store_transition(raw_obs=obs, state=state, action=actions, reward=reward,
                                    raw_obs_=obs_, state_=state_, done=done)
            obs = obs_

            if total_step % 1000 == 0 and not args.evaluate:
                maddpg_agents.learn(memory)

            score += step_reward.sum()
            episode_energy.append(step_energy)
            episode_delay.append(step_delay)
            episode_workload.append(step_workload)
            episode_success_ratio.append(step_success_ratio)

            episode_step += 1
            total_step += 1

        score_history.append(score)
        eposide_avg_energy = sum(episode_energy) / len(episode_energy)
        eposide_avg_delay = sum(episode_delay) / len(episode_delay)
        eposide_avg_workload = sum(episode_workload) / len(episode_workload)
        eposide_avg_success_ratio = sum(episode_success_ratio) / len(episode_success_ratio)

        if score_history[-100:].mean() > score_best and not args.evaluate:
            score_best = score_history[-100:].mean()
            maddpg_agents.save_checkpoint()

        if i % print_interval == 0 and i > 0:
            print('episode', i, 'score {:.1f}'.format(score))
