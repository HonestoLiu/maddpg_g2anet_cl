from station import Station
from user import User
import random

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

    def allallocate_user_to_station(self) -> None:
        pass

    def init_env(self) -> None:
        self._init_stations()
        self._init_users()
        self.allocate_user_to_station()

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

    