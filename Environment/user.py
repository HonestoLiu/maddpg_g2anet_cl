from Environment import MEC_Env
import random
import queue
import math


class Task:
    def __init__(self, appear_time) -> None:
        self.appear_time = appear_time
        self.need_cycle = random.uniform(0.2, 1)  # unit: 10^9 cycle
        self.offload_size = random.uniform(2, 5)  # unit: 10^6 bit
        self.max_dealy = 0.2                      # unit: second
        
        self.to_offload = -1
        self.allocated_resource = -1


class User:
    def __init__(self, user_id: int, x: float, y: float, side_boundary: float,
                 resource: float, trans_power: float):
        self.user_id = user_id
        self.position = (x, y)
        self.side_boundary = side_boundary
        self.resource = resource
        self.trans_power = trans_power

        self.nearest_station_id = -1
        self.nearest_station_dis = float("inf")
        self.task = Task(appear_time=0)

    def step(self, appear_time) -> None:
        x = self.position[0] + random.uniform(-2, 2)
        y = self.position[1] + random.uniform(-2, 2)
        # if out, mirror back
        x = max(x, -x)
        x = min(x, 2 * self.side_boundary - x)
        y = max(y, -y)
        y = min(y, 2 * self.side_boundary - y)
        self.position = (x, y)
        self.task = Task(appear_time)

    def print(self):
        print(f"User[{self.user_id}]: position = {self.position}")