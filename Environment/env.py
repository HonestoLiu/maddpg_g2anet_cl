import random
import numpy as np
import math
from numpy import ndarray
from queue import Queue

from Environment.station import *
from Environment.user import *

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

        self.energy_cost_coefficient = 0.5

        self.stations = []
        self.users = []

    def _init_stations(self) -> None:
        station_per_side = int(self.station_num ** 0.5)
        interval = self.side_len / station_per_side
        table_x = [i * interval for i in range(station_per_side + 1)]
        table_y = [i * interval for i in range(station_per_side + 1)]
        point_x = [(table_x[i] + table_x[i+1]) / 2 for i in range(station_per_side)]
        point_y = [(table_y[i] + table_y[i+1]) / 2 for i in range(station_per_side)]
        for i in range(self.station_num):
            x = point_x[i % station_per_side]
            y = point_y[i // station_per_side]
            self.stations.append(Station(i, x, y, self.station_cpu_frequency,
                                         self.station_band_width, self.station_noise_power))
            
    def _reset_stations(self) -> None:
       for station_i in self.stations:
            station_i.in_range_users.clear()

    def _init_users(self) -> None:
        for j in range(self.user_num):
            x = random.uniform(0, self.side_len)
            y = random.uniform(0, self.side_len)
            self.users.append(User(j, x, y, self.side_len, self.user_cpu_frequency, self.user_trans_power))

    def _reset_users(self) -> None:
        for user_j in self.users:
            x = random.uniform(0, self.side_len)
            y = random.uniform(0, self.side_len)
            user_j.position = (x, y)
            self.nearest_station_id = -1
            self.nearest_station_dis = float("inf")
            user_j.curr_task = Task(0)
            user_j.local_task_queue = Queue()
            user_j.local_queue_delay = 0

    def _allocate_users_to_stations(self) -> None:
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
            self.stations[nearest_station_id].in_range_users.append(user_j)

    def init_env(self) -> None:
        self._init_stations()
        self._init_users()
        self._allocate_users_to_stations()

    def reset_env(self) -> None:
        self._reset_stations()
        self._reset_users()
        self._allocate_users_to_stations()
        
    def step(self, now_slot, slot_size, actions_pos):
        '''
        actions : [n_agents, n_actions]
        '''
        energy_cost   = 0  # 能量
        delay_time    = 0  # 延时
        workload_size = 0  # 任务量
        success_num   = 0  # 成功个数
        reward = np.zeros((self.get_agent_num(), 3))
        
        # The order of actions correspondes to the order of users in station.
        actions_pos = actions_pos.reshape(self.station_num, -1, 2)
        for i, station_i in enumerate(self.stations):
            local_users   = []
            offload_users = []
            station_energy   = 0
            station_delay    = 0
            station_workload = 0
            station_success  = 0

            for j, user_j in enumerate(station_i.in_range_users):
                task_to_offload = int(actions_pos[i][j][0])
                if task_to_offload == 0:  # local
                    # task_resource = actions_pos[i][j][1] * user_j.resource
                    task_resource = user_j.resource
                    local_users.append(user_j)
                else:  # edge
                    task_resource = actions_pos[i][j][1] * station_i.resource
                    offload_users.append(user_j)
                user_j.set_task_info(now_slot, slot_size, task_to_offload, task_resource)
            
            # get trans_rate
            offload_trans_rate = []
            band_width_avg = station_i.band_width / len(offload_users)
            for user_j in offload_users:
                ratio = user_j.trans_power * math.pow(user_j.nearest_station_dis, -4) / station_i.noise_power
                trans_rate = band_width_avg * math.log2(1 + ratio)
                offload_trans_rate.append(trans_rate)

            # energy
            for user_j in local_users:
                station_energy += self.energy_cost_coefficient * user_j.curr_task.need_cycle * (user_j.resource ** 2)

            for j, user_j in enumerate(offload_users):
                station_energy += user_j.trans_power * user_j.curr_task.workload_size / offload_trans_rate[j]
            
            # delay
            for user_j in local_users:
                local_delay = user_j.local_queue_delay
                deal_time = user_j.curr_task.need_cycle / user_j.curr_task.resource
                station_delay += (local_delay + deal_time)
            
            for j, user_j in enumerate(offload_users):
                trans_time = user_j.curr_task.workload_size / offload_trans_rate[j]
                deal_time = user_j.curr_task.need_cycle / user_j.curr_task.resource
                # TODO(liuhong): check the reason
                deal_time = min(deal_time, slot_size)
                station_delay += (trans_time + deal_time)      

            # TODO(liuhong): 极端情况排查              

            # success
            for user_j in station_i.in_range_users:
                if user_j.curr_task.planing_finish_time <= now_slot + user_j.curr_task.max_delay:
                    station_workload += user_j.curr_task.workload_size
                    station_success += 1

            for user_j in station_i.in_range_users:
                user_j.step(now_slot)

            energy_cost   += station_energy
            delay_time    += station_delay
            workload_size += station_workload
            success_num   += station_success
            reward[i][0] -= (0.1 * station_energy)
            reward[i][1] -= (0.1 * station_delay)
            reward[i][2] += (0.1 * station_workload)

        self._reset_stations()
        self._allocate_users_to_stations()
        new_obs = self.get_obs()

        energy_cost_avg = energy_cost / self.user_num
        delay_time_avg = delay_time / self.user_num
        workload_size_avg = workload_size / self.user_num
        success_rate_avg = success_num / self.user_num
        return new_obs, reward, energy_cost_avg, delay_time_avg, workload_size_avg, success_rate_avg

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
          [5]: user task workload size
          [6]: user task max delay
        And there are n users in each station, so obs_dim is n * 7.
        '''
        return self.user_num // self.station_num * 7
    
    def get_action_dim(self) -> int:
        '''
        For each user, there are two actions
          [0](discrete, 0 or 1): whether to offload task to edge service
          [1](continuous): the ratio of allocated resource
        And there are n users in each station, so action_dim is n * 2.
        '''
        return self.user_num // self.station_num * 2

    def get_obs(self) -> ndarray:
        obs = np.zeros((self.get_agent_num(), self.get_obs_dim()), dtype=np.float32)  # [4, 35]
        for i, station_i in enumerate(self.stations):
            for j, user_j in enumerate(station_i.in_range_users):
                obs[i][7 * j + 0] = user_j.resource
                obs[i][7 * j + 1] = user_j.local_queue_delay
                obs[i][7 * j + 2] = user_j.position[0]
                obs[i][7 * j + 3] = user_j.position[1]
                obs[i][7 * j + 4] = user_j.curr_task.need_cycle
                obs[i][7 * j + 5] = user_j.curr_task.workload_size
                obs[i][7 * j + 6] = user_j.curr_task.max_delay
        return obs
    
    def print(self):
        for station_i in self.stations:
            station_i.print()