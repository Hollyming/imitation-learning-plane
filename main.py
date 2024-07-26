import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from models import TransformerModel
import utils
# from utils import predict_control, rK4_step, limit_speed, get_seeker_state, proportional_navigation, acquisition, update_lines, load_data, prepare_input, prepare_input_2, load_last_n_lines, load_first_n_lines
from missile import Missile
from flight import Flight

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

def main():
    # flight_pos_file = r"C:\Users\28208\Desktop\data_eight\data\hit_false\run_3\current_flight_pos.txt"
    # missile_sim_file = r"C:\Users\28208\Desktop\data_eight\data\hit_false\run_3\missile_sim.txt"
    flight_pos_file = r".\hit_false\run_3\current_flight_pos.txt"
    missile_sim_file = r".\hit_false\run_3\missile_sim.txt"

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


if __name__ == "__main__":
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