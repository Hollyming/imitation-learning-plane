import numpy as np
import copy
from scipy import constants
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import random
import matplotlib.animation as ani
from math import sin, cos, atan, asin, sqrt
import pandas as pd
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time


class TransformerModel(nn.Module):
    def __init__(self, input_dim, action_dim, nhead=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, dim_feedforward))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(dim_feedforward, action_dim)

    def forward(self, x):
        print("x",x.shape)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)  # 将输入展平以适应嵌入层
        x = self.embedding(x)  # 应用嵌入层
        x = x.view(batch_size, seq_len, -1)  # 重塑回 (batch_size, seq_len, dim_feedforward)
        x = x + self.positional_encoding[:, :seq_len, :]  # 加上位置编码
        x = self.transformer_encoder(x)  # 应用Transformer编码器
        x = self.fc_out(x).view(batch_size, seq_len, 3)  # 最终输出重塑
        return x

def predict_control(model, state, device):
    model.to(device)  # Ensure the model is on the correct device
    state = torch.tensor(state, dtype=torch.float32).to(device)
    #state = state.unsqueeze(0)  # 添加batch维度
    model.eval()
    with torch.no_grad():
        control = model(state)
    control = control.cpu().numpy().squeeze()
    return control


def rK4_step(state, derivs, dt, params=None, time_invariant=True, t=0):
    # One step of time-invariant RK4 integration
    if params is not None:
        k1 = dt * derivs(state, params)
        k2 = dt * derivs(state + k1 / 2, params)
        k3 = dt * derivs(state + k2 / 2, params)
        k4 = dt * derivs(state + k3, params)

        return state + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    else:
        k1 = dt * derivs(state)
        k2 = dt * derivs(state + k1 / 2)
        k3 = dt * derivs(state + k2 / 2)
        k4 = dt * derivs(state + k3)

        return state + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

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


def acquisition(target_pos):
    """
    Sets initial heading and climb angle to LOS angles, better ensuring a collision
    :param target_pos: initial detected position of target
    :return: recommended initial climb angle and heading angle
    """
    Rx = target_pos[0]
    Ry = target_pos[1]
    Rz = target_pos[2]

    Rxy = np.linalg.norm([Rx, Ry])
    R = np.linalg.norm(target_pos)

    psi_rec = asin(Ry/Rxy)
    gma_rec = asin(Rz/R)
    return [psi_rec, gma_rec]


#目标和导弹在惯性坐标系下的相对位置和速度。
def get_seeker_state(target_state, missile_state):
    """
    Computes the "seeker state," relative position and velocity in inertial coordinates
    :param target_state: position, speed, and orientation of target
    :param missile_state: position, speed, and orientation of missile
    :return: the seeker state, 3 coordinates of inertial position, 3 components of inertial velocity
    """
    # define variables for readability
    # T = target
    # M = missile

    # Target position

    RTx = target_state[0]
    RTy = target_state[1]
    RTz = target_state[2]

    # Target velocity and heading
    VT = target_state[3]
    gma_t = target_state[4]
    psi_t = target_state[5]

    # Missile Position
    RMx = missile_state[0]
    RMy = missile_state[1]
    RMz = missile_state[2]

    # Missile velocity and heading
    VM = missile_state[3]
    psi_m = missile_state[4]
    gma_m = missile_state[5]

    # Rotation matri  ces from inertial to body fixed coordinates for
    # both the target and the missile
    Rie_t = np.array([[cos(gma_t), 0, -sin(gma_t)],
                      [0, 1, 0],
                      [sin(gma_t), 0, cos(gma_t)]]).dot(
        np.array([[cos(psi_t), sin(psi_t), 0],
                  [-sin(psi_t), cos(psi_t), 0],
                  [0, 0, 1]]))

    Rie_m = np.array([[cos(gma_m), 0, -sin(gma_m)],
                      [0, 1, 0],
                      [sin(gma_m), 0, cos(gma_m)]]).dot(
        np.array([[cos(psi_m), sin(psi_m), 0],
                  [-sin(psi_m), cos(psi_m), 0],
                  [0, 0, 1]]))

    #创建一个表示目标速度向量的列向量 [VT, 0, 0] 并进行转置
    # get relative velocity in inertial coordinates
    VT_inert = Rie_t.T.dot(np.array([VT, 0, 0]).T)
    VM_inert = Rie_m.T.dot(np.array([VM, 0, 0]).T)

    return np.array([RTx - RMx, RTy - RMy, RTz - RMz,
                     VT_inert[0] - VM_inert[0], VT_inert[1] - VM_inert[1], VT_inert[2] - VM_inert[2]])

def proportional_navigation(seeker_state):
    relative_pos = np.array([seeker_state[0], seeker_state[1], seeker_state[2]])
    relative_vel = np.array([seeker_state[3], seeker_state[4], seeker_state[5]])

    # 计算期望的加速度
    norm_relative_pos = np.linalg.norm(relative_pos)

    # 防止除以零
    if norm_relative_pos == 0:
        raise ValueError("Relative position norm is zero, cannot compute desired acceleration.")

    desired_acceleration = 5 * np.cross(relative_vel, np.cross(relative_pos, relative_vel)) / norm_relative_pos ** 2

    return desired_acceleration

#设置初始条件，运行模拟循环，直到导弹命中目标或条件不满足。返回目标和导弹的轨迹以及是否命中的标志。
def limit_speed(state, max_speed):
    """
    Limit the speed of the entity to the specified maximum speed.
    :param state: The current state of the entity (x, y, z, velocity, theta, psi).
    :param max_speed: The maximum allowed speed.
    :return: The updated state with limited speed.
    """
    speed = state[3]
    if speed > max_speed:
        factor = max_speed / speed
        state[3] *= factor
    return state

def update_lines(num, target_data, missile_data, line_target, line_missile):
    line_target.set_data([target_data[:num, 0], target_data[:num,1]])
    line_target.set_3d_properties(target_data[:num, 2])
    line_missile.set_data([missile_data[:num, 0], missile_data[:num,1]])
    line_missile.set_3d_properties(missile_data[:num, 2])
    return line_target, line_missile

def prediction(mode='predict', state_input=None):
    model_path = r'C:\Users\28208\Desktop\八院\fight\pn-guidance-master\result\BC_model_Transformer_25.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if state_input is None:
        raise ValueError("For prediction mode, state_input must be provided")
    #model = TransformerModel(1200, 300).to(device)
    model = TransformerModel(input_dim=12, action_dim=3).to(device)

    #model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))

    #print(f'加载模型完毕')
    control_output = predict_control(model, state_input, device)
    #print(f'Input state: {state_input}')
    print(f'Predicted control: {control_output}')
    return control_output

def load_data(file_path, num_lines=100):
    data = np.loadtxt(file_path, dtype=np.float32, delimiter=' ')
    return data[:num_lines]

def prepare_input(flight_pos_file, missile_sim_file):
    flight_pos = load_data(flight_pos_file)
    missile_sim = load_data(missile_sim_file)
    combined_data = np.hstack((flight_pos, missile_sim))
    combined_data = combined_data.reshape(1, 100, 12)
    return combined_data

def prepare_input_2(flight_pos, missile_sim):
    flight_pos = np.array(flight_pos, dtype=np.float32)
    missile_sim = np.array(missile_sim, dtype=np.float32)
    combined_data = np.hstack((flight_pos, missile_sim))
    combined_data = combined_data.reshape(1, len(flight_pos), 12)
    return combined_data

def load_last_n_lines(file_path, num_lines=100):
    with open(file_path, 'r') as file:
        lines = file.readlines()[-num_lines:]
    data = np.array([list(map(float, line.split())) for line in lines], dtype=np.float32)
    return data

def load_first_n_lines(file_path, num_lines=1):
    with open(file_path, 'r') as file:
        lines = file.readlines()[:num_lines]
    data = np.array([list(map(float, line.split())) for line in lines], dtype=np.float32)
    return data

def main():


    flight_pos_file = r"C:\Users\28208\Desktop\data_eight\data\hit_false\run_3\current_flight_pos.txt"
    missile_sim_file = r"C:\Users\28208\Desktop\data_eight\data\hit_false\run_3\missile_sim.txt"

    state_input = prepare_input(flight_pos_file, missile_sim_file)
    print("Predicted state_input output shape:", state_input.shape)

    control_output = prediction(state_input=state_input)
    #control_output = np.squeeze(control_output)  # 确保控制输出的形状是 [100, 3]

    # 初始化初始状态

    initial_flight_state = load_first_n_lines(flight_pos_file, 1)[0]
    initial_missile_state = load_first_n_lines(missile_sim_file, 1)[0]

    current_flight_pos = [initial_flight_state]
    missile_sim = [initial_missile_state]

    seeker_state_0 = get_seeker_state(initial_flight_state, initial_missile_state)
    seeker_state = seeker_state_0
    iteration_count = 0
    distance = 500
    eps = 20
    hit = False

    flight = Flight(x=initial_flight_state[0], y=initial_flight_state[1], z=initial_flight_state[2],
                    vel=initial_flight_state[3], theta=initial_flight_state[4], phi=initial_flight_state[5],
                    delta_t=0.01, gamma=0)
    missile = Missile(x=initial_missile_state[0], y=initial_missile_state[1], z=initial_missile_state[2],
                      vel=initial_missile_state[3], phi=initial_missile_state[4], theta=initial_missile_state[5],
                      delta_t=0.01)

    missile_accel = missile.missile_accel

    while distance > eps:


        fly = []
        miss = []
        exit_loop = False  # 添加标志变量

        for t in range(100):
            iteration_count += 1

            x_overload, z_overload, gamma = control_output[t]
            flight_state = flight.basic_movement(x_overload=x_overload, z_overload=z_overload, gamma=gamma)
            commanded_accel = proportional_navigation(get_seeker_state(flight_state, [missile.x, missile.y, missile.z, missile.vel, missile.phi, missile.theta]))
            missile_state = rK4_step([missile.x, missile.y, missile.z, missile.vel, missile.phi, missile.theta], missile.missile_accel, missile.delta_t, params=commanded_accel)
            missile_state = limit_speed(missile_state, 500)
            missile.x, missile.y, missile.z, missile.vel, missile.phi, missile.theta = missile_state

            current_flight_pos.append(flight_state)
            missile_sim.append(missile_state)

            fly.append(flight_state)
            miss.append(missile_state)

            distance = np.linalg.norm([flight.x - missile.x, flight.y - missile.y, flight.z - missile.z])
            print("flight:", flight.vel)
            print("missile:", missile.vel)
            print("distance", distance)
            if distance > 20000:
                hit = False
                exit_loop = True  # 设置标志变量
                break
            if distance <= eps:
                hit = True
                exit_loop = True  # 设置标志变量
                break

        if exit_loop:  # 检查标志变量
            break


        state_input = prepare_input_2(fly, miss)
        control_output = prediction(state_input=state_input)


    current_flight_pos = np.array(current_flight_pos)
    missile_sim = np.array(missile_sim)
    print("Point:", iteration_count)
    return current_flight_pos, missile_sim, hit


current_flight_pos, missile_sim, hit = main()

plot_scale_factor = 30
target_path_plot = np.vstack((current_flight_pos[::plot_scale_factor], current_flight_pos[-1]))
missile_path_plot = np.vstack((missile_sim[::plot_scale_factor], missile_sim[-1]))

print("hit = ", hit)
ANIMATE = True
if ANIMATE:
    fig = plt.figure()
    ax = fig.add_subplot( projection='3d')
    line_target, = ax.plot(current_flight_pos[0:1, 0], current_flight_pos[0:1, 1], current_flight_pos[0:1, 2], linewidth=2)
    line_missile, = ax.plot(missile_sim[0:1, 0], missile_sim[0:1, 1], missile_sim[0:1, 2], linewidth=2)

    N = len(target_path_plot)

    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)

    anim = ani.FuncAnimation(fig, update_lines, N, fargs=(target_path_plot, missile_path_plot, line_target, line_missile),
                             interval=10, blit=False, repeat_delay=2000)

    ax.set_xlim3d([min(np.min(current_flight_pos[:, 0]), np.min(missile_sim[:, 0])),
                   max(np.max(current_flight_pos[:, 0]), np.max(missile_sim[:, 0]))])

    ax.set_ylim3d([min(np.min(current_flight_pos[:, 1]), np.min(missile_sim[:, 1])),
                   max(np.max(current_flight_pos[:, 1]), np.max(missile_sim[:, 1]))])

    ax.set_zlim3d([min(np.min(current_flight_pos[:, 2]), np.min(missile_sim[:, 2])),
                   max(np.max(current_flight_pos[:, 2]), np.max(missile_sim[:, 2]))])

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Proportional Navigation and Maneuver Library Combined')

plt.show()
