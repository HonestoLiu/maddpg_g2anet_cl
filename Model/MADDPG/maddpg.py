import torch as T
import torch.nn.functional as F
from numpy import ndarray
import numpy as np

from Model.MADDPG.agent import Agent

def numpy_softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class MADDPG:
    def __init__(self, n_agents: int, actor_dims: int, critic_dims: int, n_actions: int, alpha=0.01,
                 beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='weights/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions 
        for agent_i in range(self.n_agents):
            self.agents.append(Agent(n_agents, actor_dims, critic_dims, n_actions, agent_i,
                                     alpha, beta, fc1, fc2, gamma, tau, chkpt_dir))

    def choose_action(self, observations: ndarray) -> ndarray:
        '''
        input  : [n_agents, observation_dim]
        output : [n_agents, n_actions]
        '''
        actions = []
        for i, agent_i in enumerate(self.agents):
            observation = np.expand_dims(observations[i], axis=0)
            action = agent_i.choose_action(observation)
            actions.append(action)
        return np.concatenate(actions, axis=0)

    def pos_process_action(self, actions: ndarray) -> ndarray:
        '''
        Post-process the action, including discretization and remapping.
        input  : [n_agents, n_actions]
        output : [n_agents, n_actions]
        '''
        pos_actions = actions.copy()
        odd_slice  = range(0, self.n_actions, 2)
        even_slice = range(1, self.n_actions, 2)
        discrete_action_mean = np.mean(actions[:,odd_slice], axis=1, keepdims=True)
        pos_actions[:,odd_slice] = (actions[:,odd_slice] >= discrete_action_mean).astype(actions.dtype)

        # TODO(liuhong): try softmax remapping
        # parameter_action_min = np.min(actions[:,even_slice], axis=1, keepdims=True)
        # parameter_action_max = np.max(actions[:,even_slice], axis=1, keepdims=True)
        # pos_actions[:,even_slice] = (actions[:,even_slice] - parameter_action_min) / \
        #                             (parameter_action_max - parameter_action_min)
        pos_actions[:,even_slice] = numpy_softmax(actions[:,even_slice], axis=1)
        return pos_actions
    
    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx],
                                 dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx],
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()