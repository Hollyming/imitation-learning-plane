import numpy as np
# Missile类
class Missile:
    def __init__(self, x, y, z, vel, phi, theta, delta_t):
        self.x = x
        self.y = y
        self.z = z
        self.vel = vel
        self.phi = phi
        self.theta = theta
        self.delta_t = delta_t
        # self.max_speed = 450  # 最大速度

    @staticmethod
    def proportional_navigation(seeker_state):
        relative_pos = np.array([seeker_state[0], seeker_state[1], seeker_state[2]])
        relative_vel = np.array([seeker_state[3], seeker_state[4], seeker_state[5]])

        # 计算期望的加速度
        norm_relative_pos = np.linalg.norm(relative_pos)

        # 防止除以零
        if norm_relative_pos == 0:
            raise ValueError("Relative position norm is zero, cannot compute desired acceleration.")

        desired_acceleration = 3 * np.cross(relative_vel, np.cross(relative_pos, relative_vel)) / norm_relative_pos ** 2

        return desired_acceleration

    @staticmethod
    def missile_accel(missile_state, commanded_accel):
        Vm = missile_state[3]
        psi_m = missile_state[4]
        gma_m = missile_state[5]

        Rie_m = np.array([[np.cos(gma_m), 0, -np.sin(gma_m)],
                          [0, 1, 0],
                          [np.sin(gma_m), 0, np.cos(gma_m)]]).dot(
            np.array([[np.cos(psi_m), np.sin(psi_m), 0],
                      [-np.sin(psi_m), np.cos(psi_m), 0],
                      [0, 0, 1]]))

        Vm_inert = Rie_m.T.dot(np.array([Vm, 0, 0]).T)

        ac_body_fixed = Rie_m.dot(commanded_accel.T)

        ax = ac_body_fixed[0]
        ay = ac_body_fixed[1]
        az = ac_body_fixed[2]

        return np.array([Vm_inert[0], Vm_inert[1], Vm_inert[2],
                         ax, ay / (Vm * np.cos(gma_m)), -az / Vm])
