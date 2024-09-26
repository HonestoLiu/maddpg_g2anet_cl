
class User:
    def __init__(self, user_id: int, x: float, y: float, side_boundary: float,
                 resource: float, trans_power: float):
        self.user_id = user_id
        self.position = (x, y)
        self.side_boundary = side_boundary
        self.resource = resource
        self.trans_power = trans_power

    def print(self):
        print(f"User[{self.user_id}]: position = {self.position}")