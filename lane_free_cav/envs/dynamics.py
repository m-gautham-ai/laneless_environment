import numpy as np
class VehicleModel:
    def __init__(self, spec, dt):
        self.length, self.width = spec["L"], spec["W"]
        self.max_acc = spec["max_acc"]
        self.dt = dt
        self.reset(0, 0, 0, 0)

    def reset(self, x, y, vx, vy, heading=0.0):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.heading = heading

    def step(self, accel):
        ax = np.clip(accel[0], -self.max_acc, self.max_acc)
        ay = np.clip(accel[1], -self.max_acc, self.max_acc)
        self.vx += ax * self.dt
        self.vy += ay * self.dt
        self.x  += self.vx * self.dt
        self.y  += self.vy * self.dt
        self.heading = np.arctan2(self.vy, self.vx)
    def distance_to(self, other):
        return np.hypot(self.x - other.x, self.y - other.y)
