import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.transforms import Affine2D


class OptControl:
    def __init__(self, N, x_dim, u_dim, dyn_cons, x0, xN,uN, lower_upper_bound_ux, H):
        self.N = N
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dyn_cons = dyn_cons
        self.x0 = x0
        self.xN = xN
        self.uN = uN
        self.lb_u = lower_upper_bound_ux['lb_u']
        self.ub_u = lower_upper_bound_ux['ub_u']
        self.lb_x = lower_upper_bound_ux['lb_x']
        self.ub_x = lower_upper_bound_ux['ub_x']
        self.H = H

    def solve(self, init_guess=None):
        opti = ca.Opti()

        X = opti.variable(self.x_dim, self.N + 1)
        U = opti.variable(self.u_dim, self.N + 1)

        obj = 0
        for i in range(self.N):
            obj += (U[0, i] ** 2 + U[0, i + 1] ** 2) * self.H / 2
            obj += (U[1, i] ** 2 + U[1, i + 1] ** 2) * self.H / 2
        opti.minimize(obj)

        opti.subject_to(X[:, 0] == self.x0)
        opti.subject_to(X[:, self.N] == self.xN)
        #opti.subject_to(U[:, -1] == self.uN)
        for i in range(self.N):
            xk = X[:, i]
            xkp1 = X[:, i + 1]
            uk = U[:, i]
            ukp1 = U[:, i + 1]
            dyn = self.dyn_cons(xk, xkp1, uk, ukp1)
            opti.subject_to(dyn == 0)

        opti.subject_to(opti.bounded(self.lb_u[0], U[0, :], self.ub_u[0]))
        opti.subject_to(opti.bounded(self.lb_u[1], U[1, :], self.ub_u[1]))
        for dim in range(self.x_dim):
            opti.subject_to(opti.bounded(self.lb_x[dim], X[dim, :], self.ub_x[dim]))

        if init_guess is not None:
            X_init = init_guess[:self.x_dim * (self.N + 1)].reshape((self.x_dim, self.N + 1))
            U_init = init_guess[self.x_dim * (self.N + 1):].reshape((self.u_dim, self.N + 1))
            opti.set_initial(X, X_init)
            opti.set_initial(U, U_init)
        else:
            opti.set_initial(X, 0.01 * np.ones((self.x_dim, self.N + 1)))
            opti.set_initial(U, 0.01 * np.ones((self.u_dim, self.N + 1)))

        # 修改求解器选项以取消CasADi和IPOPT的输出
        p_opts = {"expand": True}
        s_opts = {"max_iter": 10000, "print_level": 0, "sb": "yes"}  # 添加'sb': 'yes'和'print_level':0
        opti.solver('ipopt', p_opts, s_opts)

        try:
            sol = opti.solve()
            x_opt = sol.value(X).T
            u_opt = sol.value(U).T
            return x_opt, u_opt
        except RuntimeError as e:
            print("优化求解失败:", e)
            return None, None


# 定义问题的基本变量
T0 = 0.0  # 初始时刻
Tf = 20.0  # 终止时刻
N = 500 #
H = (Tf - T0) / N
X0 = np.array([1.0, 8.0, 0.0, 0.0, 0.0])  # 初始状态约束
XN = np.array([9.25, 2.0, 0.0, 0.0, np.pi / 2.0])  # 终止状态约束
U_DIM = 2  # 动作空间维度
X_DIM = 5  # 状态空间维度
UN=np.zeros(U_DIM)
LW, LF, LR, LB = 2.8, 1.0, 1.0, 1.85  # 汽车参数：轴距，前延，后延，车宽
V_MAX, V_MIN = 3.0, -2.0  # 泊车时最大最小速度
A_MAX, A_MIN = 2.0, -1.0  # 泊车时最大最小加速度
PHI_MAX, OMEGA_MAX = 0.63792, 0.63792  # 前轮最大转角和最大角速度
Nu =15
state_noise_std = 0.04
time = np.linspace(T0, Tf, N + 1)
np.random.seed(7)
# 状态转移方程
def dynamic_f_gen(Lw: float = 2.8):
    def dynamic_f(x, u):
        """状态转移方程
        Parameters
        ----------
        x : casadi.MX 或 casadi.SX
            state, shape = [dim_x,]
        u : casadi.MX 或 casadi.SX
            action, shape = [dim_u,]

        Returns
        -------
        casadi.MX 或 casadi.SX
        """
        return ca.vertcat(
            x[2] * ca.cos(x[4]),
            x[2] * ca.sin(x[4]),
            u[0],
            u[1],
            x[2] * ca.tan(x[3]) / Lw
        )

    return dynamic_f


dynamic_f = dynamic_f_gen(LW)


def dyn_cons(xk, xkp1, uk, ukp1):
    """改进欧拉状态转移约束. 返回值需要等于 0 这是一个约束."""
    return xkp1 - xk - (dynamic_f(xk, uk) + dynamic_f(xkp1, ukp1)) * H / 2.0




def visualize_results(x_traj, u_traj, Tf, N):
    """
    可视化优化控制问题的结果，包括静态图表和动画，并显示变量的上下界
    """
    time = np.linspace(T0, Tf, N)

    # 状态变量的上下界
    state_bounds = {
        'v (velocity)': (V_MIN, V_MAX),
        'phi': (-PHI_MAX, PHI_MAX)

    }

    # 绘制状态变量
    fig1, axs1 = plt.subplots(X_DIM, 1, figsize=(10, 15), sharex=True)
    state_labels = ['x (position X)', 'y (position Y)', 'v (velocity)', 'phi', 'theta']
    for i in range(X_DIM):
        axs1[i].plot(time, x_traj[:, i], label=state_labels[i], color='blue')
        axs1[i].set_ylabel(state_labels[i])
        axs1[i].grid(True)
        axs1[i].legend(loc='upper right')

        # 如果该状态变量有上下界，绘制边界
        label = state_labels[i]
        if label in state_bounds:
            lower, upper = state_bounds[label]
            axs1[i].axhline(y=upper, color='red', linestyle='--', label='Upper Bound')
            axs1[i].axhline(y=lower, color='green', linestyle='--', label='Lower Bound')
            axs1[i].legend(loc='upper right')

    axs1[-1].set_xlabel('t (s)')
    fig1.suptitle('State Trajectories with Boundaries')
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 控制变量的上下界
    control_bounds = {
        'a (Acceleration)': (A_MIN, A_MAX),
        'omega (Steering Rate)': (-OMEGA_MAX, OMEGA_MAX)
    }

    # 绘制动作变量
    fig2, axs2 = plt.subplots(U_DIM, 1, figsize=(10, 8), sharex=True)
    control_labels = ['a (Acceleration)', 'omega (Steering Rate)']
    for i in range(U_DIM):
        axs2[i].plot(time[0:np.size(u_traj,0)], u_traj[:, i], label=control_labels[i], color='orange')
        axs2[i].set_ylabel(control_labels[i])
        axs2[i].grid(True)
        axs2[i].legend(loc='upper right')

        # 如果该控制变量有上下界，绘制边界
        label = control_labels[i]
        if label in control_bounds:
            lower, upper = control_bounds[label]
            axs2[i].axhline(y=upper, color='red', linestyle='--', label='Upper Bound')
            axs2[i].axhline(y=lower, color='green', linestyle='--', label='Lower Bound')
            axs2[i].legend(loc='upper right')

    axs2[-1].set_xlabel('t (s)')
    fig2.suptitle('Control Inputs with Boundaries')
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 绘制汽车在二维空间中的轨迹
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(x_traj[:, 0], x_traj[:, 1], 'b-', label='Trajectory')
    ax3.plot(X0[0], X0[1], 'go', label='Start')
    ax3.plot(XN[0], XN[1], 'ro', label='Goal')
    ax3.set_xlabel('Position X')
    ax3.set_ylabel('Position Y')
    ax3.set_title('Vehicle Trajectory in 2D Space')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')  # 保持比例

    # 绘制方向角随时间的变化
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(time, x_traj[:, 4], label='Theta (rad)', color='purple')
    ax4.set_xlabel('t (s)')
    ax4.set_ylabel('θ (rad)')
    ax4.set_title('Theta over Time')
    ax4.grid(True)
    ax4.legend()

    plt.show()

    # 动画部分
    fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
    ax_anim.plot(x_traj[:, 0], x_traj[:, 1], 'b-', label='Trajectory')
    ax_anim.plot(X0[0], X0[1], 'go', label='Start')
    ax_anim.plot(XN[0], XN[1], 'ro', label='Goal')
    ax_anim.set_xlabel('Position X')
    ax_anim.set_ylabel('Position Y')
    ax_anim.set_title('Vehicle Trajectory Animation')
    ax_anim.legend()
    ax_anim.grid(True)

    # 计算轨迹范围，并根据车辆尺寸调整坐标轴范围
    vehicle_length = LW + LF + LR  # 车辆长度
    vehicle_width = LB  # 车辆宽度
    min_x = np.min(x_traj[:, 0]) - vehicle_length
    max_x = np.max(x_traj[:, 0]) + vehicle_length
    min_y = np.min(x_traj[:, 1]) - vehicle_width
    max_y = np.max(x_traj[:, 1]) + vehicle_width
    ax_anim.set_xlim(min_x, max_x)
    ax_anim.set_ylim(min_y, max_y)
    ax_anim.axis('equal')  # 保持比例

    # 车辆的表示
    car_patch = patches.Rectangle((-vehicle_length / 2, -vehicle_width / 2), vehicle_length, vehicle_width,
                                  linewidth=1, edgecolor='k', facecolor='r', alpha=0.5)
    ax_anim.add_patch(car_patch)

    def init():
        car_patch.set_xy((-vehicle_length / 2, -vehicle_width / 2))
        car_patch.set_transform(Affine2D().rotate_deg(0) + ax_anim.transData)
        return car_patch,

    def animate_func(i):
        if i >= N:
            i = N - 1
        # 更新车辆位置和方向
        x = x_traj[i, 0]
        y = x_traj[i, 1]
        theta = x_traj[i, 4]
        # 创建一个新的变换
        trans = Affine2D().rotate_deg(np.degrees(theta)).translate(x, y) + ax_anim.transData
        car_patch.set_transform(trans)
        return car_patch,

    anim = animation.FuncAnimation(fig_anim, animate_func, init_func=init,
                                   frames=N, interval=100, blit=True)

    # 保存动画为GIF（可选）
    anim.save('vehicle_trajectory_noised.gif', writer='pillow')

# 容许控制与状态，状态量和动作量的约束
lower_upper_bound_ux = {
    "lb_u": np.array([A_MIN, -OMEGA_MAX]),
    "ub_u": np.array([A_MAX, OMEGA_MAX]),
    "lb_x": np.array([-ca.inf, -ca.inf, V_MIN, -PHI_MAX, -ca.inf]),
    "ub_x": np.array([ca.inf, ca.inf, V_MAX, PHI_MAX, ca.inf]),
}
e_end = np.linalg.norm(X0[:2]-XN[:2])
tolerance = 1e-2  # 状态接近目标的容差
#x0_guess = 0.01 * np.ones((N + 1) * (U_DIM + X_DIM))  # 初始猜测值

flag = 1
iteration = 0  # 迭代次数计数器
max_iterations = 1000  # 最大迭代次数，防止无限循环
N_=np.copy(N)
X0_=np.copy(X0)
H_=np.copy(H)
while e_end > tolerance and iteration < max_iterations:
    iteration += 1
    # 初始化 OptControl
    opt = OptControl(
        N=N_,
        x_dim=X_DIM,
        u_dim=U_DIM,
        dyn_cons=dyn_cons,
        x0=X0_,
        xN=XN,
        uN=UN,
        lower_upper_bound_ux=lower_upper_bound_ux,
        H=H_
    )

    # 求解最优控制问题
    #xks, uks = opt.solve(init_guess=x0_guess)
    xks, uks = opt.solve()
    if xks is None or uks is None:
        print(f"在第 {iteration} 次迭代中未能找到最优解。")
        break

    # 输出当前迭代的控制输入

    print(f"迭代 {iteration}:")
    print(f"控制输入 U0 = 加速度: {uks[0,0]:.4f} m/s², 角速度: {uks[1,0]:.4f} rad/s")
    nu=min(Nu,xks.shape[0])
    for i in range(nu):
        x_now = xks[i, :]
        if i==nu-1:
            delta_x =dynamic_f(x_now, uks[i, :])  * H_
        else:
            delta_x = (dynamic_f(x_now, uks[i,:]) + dynamic_f(xks[i+1,:], uks[i+1,:])) * H_ / 2.0
        x_next = x_now + delta_x
        noise = 0.03 * x_now * np.random.normal(0, state_noise_std, size=x_now.shape)
        x_next += noise
        x_next = x_next.T
        if flag:
            final_x = xks[:2, :]
            final_u = uks[0, :]
            flag = 0
        else:
            final_x = np.vstack((final_x, x_next))
            final_u = np.vstack((final_u, uks[i, :]))
        X0_ = x_next.T
        e_end = np.linalg.norm(X0_[:2] - XN[:2])
        if e_end < tolerance:
            break

    H_ = (Tf - T0) / N_
    N_-=i+1


    guess_x=xks[nu:,:].reshape(-1,1)
    guess_u=uks[nu:,:].reshape(-1,1)
    #x0_guess=np.vstack((guess_x, guess_u))
    print(f"当前误差 e_end = {e_end:.4f}\n")

visualize_results(final_x, final_u, Tf, np.size(final_x,0))
print("Gif graph saved successfully!")