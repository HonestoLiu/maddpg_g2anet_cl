import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        self.mem_cntr = 0
        self.mem_size = max_size
        self.critic_dim = critic_dims
        self.actor_dims = actor_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.batch_size = batch_size        

        self.init_memory()

    def init_memory(self):
        self.state_memory     = np.zeros((self.mem_size, self.critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, self.critic_dims))
        self.reward_memory    = np.zeros((self.mem_size, self.n_agents))
        self.terminal_memory  = np.zeros((self.mem_size, self.n_agents), dtype=bool)
    
        self.actor_state_memory     = []
        self.actor_new_state_memory = []
        self.actor_action_memory    = []

        for agent_i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[agent_i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[agent_i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        '''
        raw_obs/raw_obs_ : [n_agent, actor_dim]
        state/state_     : [critic_dim, ]
        action           : [n_agent, n_action]
        reward           : [n_agent, ]
        done             : [n_agent, ]
        '''
        index = self.mem_cntr % self.mem_size

        self.state_memory[index]     = state
        self.new_state_memory[index] = state_
        self.reward_memory[index]    = reward
        self.terminal_memory[index]  = done

        for agent_i in range(self.n_agents):
            self.actor_state_memory[agent_i][index]     = raw_obs[agent_i]
            self.actor_new_state_memory[agent_i][index] = raw_obs_[agent_i]
            self.actor_action_memory[agent_i][index]    = action[agent_i]

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states   = self.state_memory[batch]
        states_  = self.new_state_memory[batch]
        rewards  = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states     = []
        actor_new_states = []
        actions          = []
        
        for agent_i in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_i][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_i][batch])
            actions.append(self.actor_action_memory[agent_i][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
