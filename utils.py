import numpy as np
import torch
from math import sin, cos, atan, asin, sqrt
# 工具函数
def predict_control(model, state, device):
    model.to(device)  # Ensure the model is on the correct device
    state = torch.tensor(state, dtype=torch.float32).to(device)
    #state = state.unsqueeze(0)  # 添加batch维度
    model.eval()
    with torch.no_grad():
        control = model(state)
    control = control.cpu().numpy().squeeze()
    return control

# Runge-Kutta 4阶积分
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
        factor = max_speed / speed#计算因子调整速度
        state[3] *= factor
    return state

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
    # 拆分为相对位置和速度
    relative_pos = np.array([seeker_state[0], seeker_state[1], seeker_state[2]])
    relative_vel = np.array([seeker_state[3], seeker_state[4], seeker_state[5]])
    # 计算相对位置向量的模长
    norm_relative_pos = np.linalg.norm(relative_pos)
    # 防止除以零
    if norm_relative_pos == 0:
        raise ValueError("Relative position norm is zero, cannot compute desired acceleration.")
    #计算期望加速度
    desired_acceleration = 5 * np.cross(relative_vel, np.cross(relative_pos, relative_vel)) / norm_relative_pos ** 2
    return desired_acceleration

# 计算目标的初始瞄准角度
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

def update_lines(num, target_data, missile_data, line_target, line_missile):
    """
    更新目标线和导弹线的数据。
    
    这个函数用于在3D图形中更新目标和导弹的位置。它通过修改line_target和line_missile对象的数据，
    来反映target_data和missile_data中最新的位置信息。只更新前num个数据点，以提高更新效率或应对大量数据的情况。
    
    参数:
    num -- 更新的数据点数量。
    target_data -- 目标的位置数据，包括x、y、z坐标。
    missile_data -- 导弹的位置数据，包括x、y、z坐标。
    line_target -- 表示目标的线对象。
    line_missile -- 表示导弹的线对象。
    
    返回:
    line_target -- 更新后的目标线对象。
    line_missile -- 更新后的导弹线对象。
    """
    # 更新目标线的数据
    line_target.set_data([target_data[:num, 0], target_data[:num, 1]])
    line_target.set_3d_properties(target_data[:num, 2])
    
    # 更新导弹线的数据
    line_missile.set_data([missile_data[:num, 0], missile_data[:num, 1]])
    line_missile.set_3d_properties(missile_data[:num, 2])
    
    return line_target, line_missile

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