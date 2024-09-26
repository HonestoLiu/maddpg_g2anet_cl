from user import User

class Station:
    def __init__(self, station_id: int, x: float, y: float , resource: float,
                 band_width: float, noise_power: float) -> None:
        self.station_id = station_id
        self.position = (x, y)
        self.resource = resource
        self.band_width = band_width
        self.noise_power = noise_power
        self.in_range_users = []

    def append_user(self, user: User):
        self.in_range_users.append(user)

    def print(self):
        print(f"Station[{self.station_id}]: position = {self.position}")