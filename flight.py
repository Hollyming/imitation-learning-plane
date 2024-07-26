import numpy as np
from scipy import constants

class Flight(object):
    def __init__(self, x, y, z, vel, theta, phi, delta_t, gamma):
        self.x = x
        self.y = y
        self.z = z
        self.vel = vel
        self.phi = phi
        self.theta = theta
        self.delta_t = delta_t
        self.tau = 100 * self.delta_t
        self.gamma = gamma
        self.max_gamma_change_rate = 0.01
        self.max_x_overload = 0.5
        self.min_x_overload = 0
        self.max_z_overload = 0.5
        self.min_z_overload = 0
        self.min_speed = 50
        # self.max_speed = 280  # 最大速度
        self.traj_list = []

    def basic_movement(self, x_overload, z_overload, gamma):
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        if self.vel == 0 or cos_theta == 0:
            raise ValueError("self.vel and np.cos(self.theta) cannot be 0 to avoid division by zero")

        delta_x = self.vel * cos_theta * np.cos(self.phi) * self.delta_t
        delta_y = self.vel * cos_theta * np.sin(self.phi) * self.delta_t
        delta_z = self.vel * sin_theta * self.delta_t

        delta_vel = constants.g * (x_overload - sin_theta) * self.delta_t
        delta_theta = constants.g * (z_overload * np.cos(gamma) - cos_theta) / (self.vel) * self.delta_t
        delta_phi = constants.g * z_overload * np.sin(gamma) / (self.vel * cos_theta) * self.delta_t

        self.x += delta_x
        self.y += delta_y
        self.z += delta_z
        self.vel += delta_vel
        self.theta += delta_theta
        self.phi += delta_phi
        # 限制速度最小是50，最大是300
        if self.vel < 50:
            self.vel = 50
        elif self.vel > 300:
            self.vel = 300

        return self.x, self.y, self.z, self.vel, self.theta, self.phi
