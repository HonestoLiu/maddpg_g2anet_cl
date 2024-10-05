class Station:
    def __init__(self, station_id: int, x: float, y: float , resource: float,
                 band_width: float, noise_power: float) -> None:
        self.station_id = station_id
        self.position = (x, y)
        self.resource = resource
        self.band_width = band_width
        self.noise_power = noise_power
        self.in_range_users = []

    def print(self):
        print(f"Station[{self.station_id}]: position = {self.position}")
        for user in self.in_range_users:
            user.print()