import matplotlib.pyplot as plt
import matplotlib.animation as animation
from trajectory_generate import trajectory_generate
from Vehicle_Dynamic import VehicleDynamic
from AL_iLQR_tracking import *

# 定义问题的基本变量
T0 = 0.0  # 初始时刻
Tf = 20.0  # 终止时刻
H = 0.2
N = int((Tf - T0) / H) + 1

X0 = np.array([1.0, 8.0, 0.0, 0.0, 0.0])  # 初始状态约束
XN = np.array([9.25, 2.0, 0.0, 0.0, np.pi / 2.0])  # 终止状态约束

U_DIM = 2  # 动作空间维度
X_DIM = 5  # 状态空间维度
UN = np.zeros(U_DIM)
LW, LF, LR, LB = 2.8, 1.0, 1.0, 1.85  # 汽车参数：轴距，前延，后延，车宽
V_MAX, V_MIN = 3.0, -2.0  # 速度限制
A_MAX, A_MIN = 2.0, -1.0  # 加速度限制
PHI_MAX, OMEGA_MAX = 0.63792, 0.63792  # 转角和角速度限制

# 计算MSE的函数
def compute_mse(desired_states, actual_states):
    error = desired_states - actual_states
    mse = np.mean(np.square(error), axis=0)  # 对每个状态变量计算MSE
    return mse

# 绘制静态图函数
def plot_static_figures(desired_states, actual_states, desired_control, actual_control):
    time_steps = np.arange(desired_states.shape[0])

    # 绘制状态变量随时间变化
    ncols_state = 2
    nrows_state = int(np.ceil(X_DIM / ncols_state))
    fig_states, axs_states = plt.subplots(nrows_state, ncols_state, figsize=(15, 5 * nrows_state))
    axs_states = axs_states.flatten()  # 将二维数组展平成一维

    for i in range(X_DIM):
        axs_states[i].plot(time_steps, desired_states[:, i], label=f'Desired State {i + 1}', linestyle='--')
        axs_states[i].plot(time_steps, actual_states[:, i], label=f'Actual State {i + 1}')
        axs_states[i].set_xlabel('Time Step')
        axs_states[i].set_ylabel(f'State {i + 1}')
        axs_states[i].set_title(f'State Variable {i + 1} Over Time')
        axs_states[i].legend()
        axs_states[i].grid(True)

    # 如果子图数量多于变量数量，隐藏多余的子图
    for j in range(X_DIM, len(axs_states)):
        fig_states.delaxes(axs_states[j])

    plt.tight_layout()
    plt.show()

    # 绘制控制输入随时间变化
    ncols_control = 2
    nrows_control = int(np.ceil(U_DIM / ncols_control))
    fig_controls, axs_controls = plt.subplots(nrows_control, ncols_control, figsize=(15, 5 * nrows_control))
    axs_controls = axs_controls.flatten()

    for i in range(U_DIM):
        axs_controls[i].plot(time_steps, desired_control[:, i], label=f'Desired Control {i + 1}', linestyle='--')
        axs_controls[i].plot(time_steps, actual_control[:, i], label=f'Actual Control {i + 1}')
        axs_controls[i].set_xlabel('Time Step')
        axs_controls[i].set_ylabel(f'Control {i + 1}')
        axs_controls[i].set_title(f'Control Input {i + 1} Over Time')
        axs_controls[i].legend()
        axs_controls[i].grid(True)

    # 如果子图数量多于变量数量，隐藏多余的子图
    for j in range(U_DIM, len(axs_controls)):
        fig_controls.delaxes(axs_controls[j])

    plt.tight_layout()
    plt.show()

# 创建动画函数
def create_animation(desired_states, actual_states, mse_history, interval=200):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # 子图1: 轨迹动画
    ax_traj = axs[0]
    ax_traj.set_xlim(min(np.min(desired_states[:, 0]), np.min(actual_states[:, 0])) - 1,
                     max(np.max(desired_states[:, 0]), np.max(actual_states[:, 0])) + 1)
    ax_traj.set_ylim(min(np.min(desired_states[:, 1]), np.min(actual_states[:, 1])) - 1,
                     max(np.max(desired_states[:, 1]), np.max(actual_states[:, 1])) + 1)
    ax_traj.set_xlabel('X Position')
    ax_traj.set_ylabel('Y Position')
    ax_traj.set_title('Tracking Process Animation')

    desired_line, = ax_traj.plot([], [], 'g--', label='Desired Trajectory')
    actual_line, = ax_traj.plot([], [], 'b-', label='Actual Trajectory')
    desired_point, = ax_traj.plot([], [], 'go')
    actual_point, = ax_traj.plot([], [], 'bo')
    ax_traj.legend()
    ax_traj.grid(True)

    # 子图2: MSE随时间变化
    ax_mse = axs[1]
    ax_mse.set_xlim(0, len(mse_history))
    ax_mse.set_ylim(0, np.max(mse_history) * 1.1)
    ax_mse.set_xlabel('Time Step')
    ax_mse.set_ylabel('MSE')
    ax_mse.set_title('Mean Squared Error Over Time')
    mse_line, = ax_mse.plot([], [], 'r-')
    ax_mse.grid(True)

    plt.tight_layout()

    def init():
        # 初始化轨迹动画
        desired_line.set_data([], [])
        actual_line.set_data([], [])
        desired_point.set_data([], [])
        actual_point.set_data([], [])
        # 初始化MSE曲线
        mse_line.set_data([], [])
        return desired_line, actual_line, desired_point, actual_point, mse_line

    def animate(i):
        # 更新轨迹动画
        desired_line.set_data(desired_states[:i, 0], desired_states[:i, 1])
        actual_line.set_data(actual_states[:i, 0], actual_states[:i, 1])
        desired_point.set_data(desired_states[i, 0], desired_states[i, 1])
        actual_point.set_data(actual_states[i, 0], actual_states[i, 1])

        # 更新MSE曲线
        mse_line.set_data(np.arange(i + 1), mse_history[:i + 1])

        return desired_line, actual_line, desired_point, actual_point, mse_line

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(desired_states), interval=interval, blit=True, repeat=False)

    # 保存动画为GIF
    ani.save('trajectory_tracking.gif', writer='pillow')
    print("动画已保存为 'trajectory_tracking.gif'")


# 可视化跟踪过程及其他绘图
def visualize_tracking(desired_states, actual_states, desired_control, actual_control, mse_history):
    # 绘制静态图
    plot_static_figures(desired_states, actual_states, desired_control, actual_control)

    # 创建并保存动画
    create_animation(desired_states, actual_states, mse_history)

# 主流程
def main():
    # 定义符号变量
    x, y, v, phi, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('v'), ca.SX.sym('phi'), ca.SX.sym('theta')
    a, omega = ca.SX.sym('a'), ca.SX.sym('omega')
    state = ca.vertcat(x, y, v, phi, theta)
    control = ca.vertcat(a, omega)
    ineq_constraints = []
    # 状态约束
    ineq_constraints.append(v - V_MAX)  # v <= V_MAX
    ineq_constraints.append(V_MIN - v)  # v >= V_MIN
    ineq_constraints.append(phi - PHI_MAX)  # phi <= PHI_MAX
    ineq_constraints.append(-phi - PHI_MAX)  # phi >= -PHI_MAX
    # 控制约束
    ineq_constraints.append(a - A_MAX)  # a <= A_MAX
    ineq_constraints.append(A_MIN - a)  # a >= A_MIN
    ineq_constraints.append(omega - OMEGA_MAX)  # omega <= OMEGA_MAX
    ineq_constraints.append(-omega - OMEGA_MAX)  # omega >= -OMEGA_MAX

    # 等式约束
    eq_constraints = []
    # 如有需要，请在这里添加等式约束

    # 第一次轨迹生成（无噪声）
    Q_initial = np.eye(X_DIM) * 0.00
    R_initial = np.eye(U_DIM) * 1
    Qn_initial = np.eye(X_DIM) * 100
    np.random.seed(7)
    u_initial_guess = np.random.uniform(low=[A_MIN, -OMEGA_MAX],
                                        high=[A_MAX, OMEGA_MAX],
                                        size=(N, U_DIM)) * 0.1
    Desired_State, Desired_Control = trajectory_generate(
        VehicleDynamic(LW, H),
        CostFunc(XN, Q_initial, R_initial, Qn_initial, N),
        ConstraintFunction(eq_constraints, ineq_constraints, state, control),
        X0, XN, u_initial_guess, N, False
    )

    # 添加噪声后的车辆模型
    std_noise = np.array([0.5, 0.5, 0.2, 0.02, 0.02])
    scale_noise = 0.1
    Vehicle_noised = VehicleDynamicNoised(LW, H, scale_noise, std_noise)

    # 第二次轨迹生成（有噪声）
    Q = np.eye(X_DIM) * 1
    R = np.eye(U_DIM) * 1
    Qn = np.eye(X_DIM) * 1
    # 记录开始时间
    import time
    time_start = time.time()
    Actual_State, Actual_Control = trajectory_tracking(
        Vehicle_noised,
        CostFunc_tacking(XN, Q , R, Qn,N,ref_x=Desired_State,ref_u=Desired_Control),
        ConstraintFunction(eq_constraints, ineq_constraints, state, control),
        X0, XN, Desired_Control, N,Desired_State,Desired_Control
    )

    # 记录结束时间并打印耗时
    time_end = time.time()
    print(f"Trajectory tracking took {time_end - time_start:.2f} seconds.")

    # 计算MSE
    mse = compute_mse(Desired_State, Actual_State)
    print("Mean Squared Error (MSE) per state variable:", mse)

    # 计算MSE随时间变化（逐步计算）
    mse_history = np.mean((Desired_State[:len(Actual_State)] - Actual_State) ** 2, axis=1)

    # 可视化跟踪过程及其他绘图
    visualize_tracking(Desired_State, Actual_State, Desired_Control, Actual_Control, mse_history)

if __name__ == "__main__":
    main()
