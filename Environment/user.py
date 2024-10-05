import random
from queue import Queue

class Task:
    def __init__(self, appear_time: int):
        self.appear_time = appear_time
        self.need_cycle = random.uniform(0.2, 1.2)   # unit: 10^9 cycle
        self.workload_size = random.uniform(2, 5)  # unit: 10^6 bit
        self.max_delay = 0.2                       # unit: second
        
        self.to_offload = 0
        self.resource = 0
        self.planing_finish_time = 0

class User:
    def __init__(self, user_id: int, x: float, y: float, side_boundary: float,
                 resource: float, trans_power: float):
        self.user_id = user_id
        self.position = (x, y)
        self.side_boundary = side_boundary
        self.resource = resource  # cpu frequency
        self.trans_power = trans_power

        self.nearest_station_id = -1
        self.nearest_station_dis = float("inf")

        # task info
        self.curr_task = Task(0)
        self.local_task_queue = Queue()
        self.local_queue_delay = 0

    def set_task_info(self, now_slot, slot_size, task_to_offload, task_resource):
        self.curr_task.to_offload = task_to_offload
        self.curr_task.resource = task_resource
        if task_to_offload == 0:  # local
            deal_time = self.curr_task.need_cycle / self.resource
            self.curr_task.planing_finish_time = now_slot - slot_size + self.local_queue_delay + deal_time
        else:  # edge
            self.curr_task.planing_finish_time = now_slot

    def step(self, now_slot):
        # update position, mirror it back if out
        x = self.position[0] + random.uniform(-2, 2)
        y = self.position[1] + random.uniform(-2, 2)
        x = max(x, -x)
        x = min(x, 2 * self.side_boundary - x)
        y = max(y, -y)
        y = min(y, 2 * self.side_boundary - y)
        self.position = (x, y)

        # put curr_task into the queue
        if self.curr_task.to_offload == 0:
            self.local_task_queue.put(self.curr_task)

        # remove finished task
        unfinish_tasks = []
        while not self.local_task_queue.empty():
            task = self.local_task_queue.get()
            if task.planing_finish_time > now_slot:
                self.local_queue_delay = task.planing_finish_time - now_slot
                unfinish_tasks.append(task)
            else:
                self.local_queue_delay = 0
        
        for task in unfinish_tasks:
            self.local_task_queue.put(task)

        # generate a new curr_task
        self.curr_task = Task(now_slot)

    def print(self):
        print(f"    User[{self.user_id:2}]: position = ({self.position[0]:4.2f}" +
              f", {self.position[1]:4.2f}), station_id = {self.nearest_station_id:2}")