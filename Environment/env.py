from station import Station
from user import User
from user import Task
import random
import numpy as np
from numpy import ndarray
import math

class MEC_Env:
    def __init__(self, side_len: float, station_num: int, user_num: int, station_cpu_frequency: float,
                 station_band_width: float, station_noise_power: float, user_cpu_frequency: float,
                 user_trans_power: float) -> None:
        self.side_len = side_len
        self.station_num = station_num
        self.user_num = user_num
        self.station_cpu_frequency = station_cpu_frequency
        self.station_band_width = station_band_width
        self.station_noise_power = station_noise_power
        self.user_cpu_frequency = user_cpu_frequency
        self.user_trans_power = user_trans_power

        self.stations = []
        self.users = []

    def _init_stations(self) -> None:
        station_per_side = int(self.station_num ** 0.5)
        interval = self.side_len / station_per_side
        table_x = [i * interval for i in range(station_per_side)]
        table_y = [i * interval for i in range(station_per_side)]
        point_x = [(table_x[i] + table_x[i+1]) / 2 for i in range(station_per_side)]
        point_y = [(table_y[i] + table_y[i+1]) / 2 for i in range(station_per_side)]
        for i in range(self.station_num):
            x = point_x[i % station_per_side]
            y = point_y[i // station_per_side]
            self.stations.append(Station(i, x, y, self.station_cpu_frequency,
                                         self.station_band_width, self.station_noise_power))
    
    def _init_users(self) -> None:
        for i in range(self.user_num):
            x = random.uniform(0, self.side_len)
            y = random.uniform(0, self.side_len)
            self.users.append(User(i, x, y, self.side_len, self.user_cpu_frequency, self.user_trans_power))

    def _reset_users(self) -> None:
        for user_i in self.users:
            user_i.position[0] = random.uniform(0, self.side_len)
            user_i.position[1] = random.uniform(0, self.side_len)
            user_i.task = Task(0)

    def allallocate_users_to_stations(self) -> None:
        for station_i in self.stations:
            station_i.in_range_users.clear()

        user_num_per_station = self.user_num // self.station_num
        for user_j in self.users:
            nearest_station_id = -1
            nearest_station_dis = float("inf")
            x1 = user_j.position[0]
            y1 = user_j.position[1]
            for i, station_i in enumerate(self.stations):
                station_is_full = len(station_i.in_range_users) == user_num_per_station
                x2 = station_i.position[0]
                y2 = station_i.position[1]
                distance = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
                if distance < nearest_station_dis and not station_is_full:
                    nearest_station_dis = distance
                    nearest_station_id = i
            user_j.nearest_station_dis = nearest_station_dis
            user_j.nearest_station_id = nearest_station_id
            self.stations[nearest_station_id].append_user(user_j)

    def init_env(self) -> None:
        self._init_stations()
        self._init_users()
        self.allocate_user_to_station()

    def reset_env(self) -> None:
        self._reset_users()
        self.allallocate_users_to_stations()
        
    def step(self, now_slot: int, actions_pos: ndarray):
        '''
        actions : [n_agents, n_actions]
        '''
        # The order of actions correspondes to the order of users in station.
        actions_pos = actions_pos.reshape(self.station_num, -1, 2)
        for i, station_i in enumerate(self.stations):
            for j, user_j in station_i.in_range_users:
                if actions_pos[i][j][0] == 0:  # local
                    user_j.task.allocated_resource = actions_pos[i][j][1] * user_j.resource
                elif actions_pos[i][j][1] == 1:  # edge
                    user_j.task.allocated_resource = actions_pos[i][j][1] * station_i.resource

        


        

    def get_agent_num(self) -> int:
        return self.station_num
    
    def get_obs_dim(self) -> int:
        '''
        For each user, there are seven observation dims
          [0]: user local resource
          [1]: user local queue delay
          [2]: user position_x
          [3]: user position_y
          [4]: user task cycle
          [5]: user task offload size
          [6]: user task max delay
        And there are n uers in each station, so obs_dim is n * 7.
        '''
        return len(self.user_num) // len(self.station_num) * 7
    
    def get_action_dim(self) -> int:
        '''
        For each user, there are two actions
          [0](discrete, 0 or 1): whether to offload task to edge service
          [1](continuous): the ratio of allocated resource
        And there are n users in each station, so action_dim is n * 2.
        '''
        return len(self.user_num) // len(self.station_num) * 2

    def get_obs(self) -> ndarray:
        obs = np.zeros((self.get_agent_num(), self.get_obs_dim()), dtype=np.float32)  # [4, 35]
        for i, station_i in enumerate(self.stations):
            for j, user_j in enumerate(station_i.in_range_users):
                obs[i][7 * j + 0] = user_j.resource
                obs[i][7 * j + 1] = user_j.queue_delay
                obs[i][7 * j + 2] = user_j.position[0]
                obs[i][7 * j + 3] = user_j.position[1]
                obs[i][7 * j + 4] = user_j.task.need_cycle
                obs[i][7 * j + 5] = user_j.task.offload_size
                obs[i][7 * j + 6] = user_j.task.max_delay
        return obs
    