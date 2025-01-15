import matplotlib.pyplot as plt
from matplotlib import animation, patches
from matplotlib.transforms import Affine2D
from Vehicle_Dynamic import VehicleDynamic
from CostFunc import *

# 定义问题的基本变量
T0 = 0.0  # 初始时刻
Tf = 20.0  # 终止时刻
H = 0.2
N = int((Tf-T0)/H)+1

X0 = np.array([1.0, 8.0, 0.0, 0.0, 0.0])  # 初始状态约束
XN = np.array([9.25, 2.0, 0.0, 0.0, np.pi / 2.0])  # 终止状态约束

U_DIM = 2  # 动作空间维度
X_DIM = 5  # 状态空间维度
UN=np.zeros(U_DIM)
LW, LF, LR, LB = 2.8, 1.0, 1.0, 1.85  # 汽车参数：轴距，前延，后延，车宽
V_MAX, V_MIN = 3.0, -2.0  # 速度限制
A_MAX, A_MIN = 2.0, -1.0  # 加速度限制
PHI_MAX, OMEGA_MAX = 0.63792, 0.63792  # 转角和角速度限制




class ConstraintFunction:
    def __init__(self,eq_Constrint:list,ineq_Constrint:list,State,Control):
        self.ineq_Constrint = ineq_Constrint
        self.eq_Constrint = eq_Constrint
        #不等式约束在前，等式约束在后
        self.Constaint=ineq_Constrint+eq_Constrint
        self.n_cons=len(self.Constaint)
        self.State = State
        self.Control = Control
        #Penalty Muti
        self.mu = ca.SX.sym("mu", self.n_cons)
        self.I_mu = ca.diag(self.mu)
        #Lagrange Muti
        self.Lambda = ca.SX.sym("lambda_eq", self.n_cons)
        self.ConstraintFunction()
        self.ALMTerm()

    def ConstraintFunction(self):
        # transform Constraint cost into max(constraint term,0)
        for i in range(len(self.ineq_Constrint)):
            self.ineq_Constrint[i] = ca.mmax(ca.vertcat(self.ineq_Constrint[i], 0))

        # the formular of constraint
        self.ConstrintCostFormular = ca.vertcat(*self.Constaint)

        # the calculate function of constraint
        self.ConstrintCalFunction = ca.Function("constraintFunction",[self.State,self.Control],[self.ConstrintCostFormular])

        Cx = ca.jacobian(self.ConstrintCostFormular,self.State)
        Cu = ca.jacobian(self.ConstrintCostFormular,self.Control)

        self.CxFunc = ca.Function("Cx",[self.State,self.Control],[Cx])
        self.CuFunc = ca.Function("Cu",[self.State,self.Control],[Cu])
    def ALMTerm(self):
        # get the Augment Langrange Term using second order Cost Term
        AugmentTerm = 0
        for i in range(len(self.ineq_Constrint)):
            AugmentTerm += 0.5*self.mu[i]*(ca.power(ca.mmax(ca.vertcat(self.Lambda[i] / self.mu[i] + self.Constaint[i], 0)), 2) - ca.power(self.Lambda[i] / self.mu[i], 2))
        for i in range(len(self.ineq_Constrint),self.n_cons):
            AugmentTerm += 0.5*self.mu[i]*(ca.power(self.Lambda[i] / self.mu[i] + self.Constaint[i], 2) - ca.power(self.Lambda[i] / self.mu[i], 2))

        cx = ca.jacobian(AugmentTerm,self.State)
        cu = ca.jacobian(AugmentTerm,self.Control)
        cxx = ca.jacobian(cx ,self.State)
        cux = ca.jacobian(cu ,self.State)
        cuu = ca.jacobian(cu ,self.Control)
        # Get the Augment Cost function
        self.AugmentFunction = ca.Function("ALM_Function", [self.State, self.Control, self.mu, self.Lambda], [AugmentTerm])
        # Get the jacobian function of Augment Cost
        self.CxFunc_ = ca.Function("Cx", [self.State, self.Control, self.mu, self.Lambda], [cx])
        self.CuFunc_ = ca.Function("Cx", [self.State, self.Control, self.mu, self.Lambda], [cu])
        # Get the Hessian function of Augment Cost
        self.CxxFunc_ = ca.Function("Cx", [self.State, self.Control, self.mu, self.Lambda], [cxx])
        self.CuxFunc_ = ca.Function("Cx", [self.State, self.Control, self.mu, self.Lambda], [cux])
        self.CuuFunc_ = ca.Function("Cx", [self.State, self.Control, self.mu, self.Lambda], [cuu])




class ALILQR:
    def __init__(self,VehicleDynamic:VehicleDynamic,CostFunc:CostFunc,ConstraintFunc:ConstraintFunction, X_start,X_goal,Nt,needPrint=True):
        '''

        @param VehicleDynamic: the dynamic transfer function
        @param CostFunc:  the cost function of vehicle system
        @param ConstraintFunc: the constraint function of the function
        @param ref_state: the reference state of vehicle
        @param step:
        '''
        self.VehicleDynamic = VehicleDynamic
        self.CostFunc = CostFunc
        self.ConstraintFunc = ConstraintFunc
        self.step = Nt
        self.ref_state = np.ones((Nt+1,X_DIM))*X_goal
        # initialize the multiplier
        self.Lambda = [ca.DM.zeros(len(self.ConstraintFunc.Constaint)) for _ in range(self.step)]
        # initialize the penality term
        self.Mu     = [1e-1 * ca.DM.ones(len(self.ConstraintFunc.Constaint)) for _ in range(self.step)]
        self.X_goal = X_goal
        self.X_start = X_start
        self.max_iter = 50
        self.line_search_beta_1 = 1e-8
        self.line_search_beta_2 = 5
        self.line_search_gamma = 0.5
        self.J_tolerance = 1e-2
        self.ConstraintTolerance = 1e-4
        # self.ConstraintTolerance = 1
        self.beta=1e-4
        self.MuFactor = 1.1
        self.MuMax = 1e8
        self.needPrint = needPrint

    def Evalueate(self,x,u):
        '''
        function : Evaluate the total trajectory
        @param x: the state of vehicle
        @param u: the control of vehicle
        @return:
        '''
        J = self.CostFunc.CalcCost(x,u)
        for i in range(self.step):
            # J += self.ConstraintFunc.AugmentPenelityFunction(x[i],u[i],self.I_mu[i],self.Lambda[i])
            J += self.ConstraintFunc.AugmentFunction(x[i], u[i], self.Mu[i], self.Lambda[i])
        return J

    def init_trajectory(self,u):
        '''
        function : roll out
        @param u: the control input of vehicle
        @return:
        '''
        x_init = []
        x_init.append(self.X_start)
        for i in range(self.step):
            x_init.append(self.VehicleDynamic.propagate(x_init[-1],u[i]))
        return x_init
    def CalcConstraintViolation(self,x,u):
        '''
        @function: calculate the Constraint of Violation
        @param x: the state input
        @param u: the control input
        @return: the Constraint Violation
        '''
        Vio = 0
        for i in range(len(u)):
            Vio += self.ConstraintFunc.ConstrintCalFunction(x[i],u[i])
        return ca.sum1(Vio)
    def CalcMaxVio(self,x,u):
        '''
        function: update the constraint violation term
        @param x:
        @param u:
        @return:
        '''
        res = -1000
        for i in range(len(u)):
            Vio = self.ConstraintFunc.ConstrintCalFunction(x[i],u[i])
            maxvio = ca.mmax(Vio)
            res = max(res,maxvio)
        return res

    def UpdatePenalityParam(self,x,u):
        '''
        function: update the langrange multiplier and penality term
        @param x:the state of vehicle
        @param u:the control of vehicle
        @return:
        '''
        for i in range(self.step):
            Vio = self.ConstraintFunc.ConstrintCalFunction(x[i],u[i])
            for j in range(len(self.ConstraintFunc.ineq_Constrint)):
                self.Lambda[i][j]=max(0,self.Mu[i][j]*Vio[j]+self.Lambda[i][j])
                self.Mu[i][j] *= self.MuFactor
            for j in range(len(self.ConstraintFunc.ineq_Constrint),len(self.ConstraintFunc.Constaint)):
                self.Lambda[i][j]+=self.Mu[i][j]*Vio[j]
                self.Mu[i][j] *= self.MuFactor

    # 正定性检查函数
    def is_pos_def(self,matrix):
        """
        检查矩阵是否为正定
        """
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    def regularized_persudo_inverse(self,mat, reg=1e-1):
        u, s, v = np.linalg.svd(mat)
        for i in range(len(s)):
            if s[i] < 0:
                s.at[i].set(0.0)
                print("Warning: inverse operator singularity{0}".format(i))
        diag_s_inv = np.diag(1. / (s + reg))
        return ca.DM(v.dot(diag_s_inv).dot(u.T))

    def BackWardPass(self,x,u):
        '''
        the BackWard Process in ILQR
        @param x: the state input
        @param u: the control input
        @return: k,d,Qu_list,Quu_list
        '''
        p = self.CostFunc.p_fun(self.X_goal ,x[-1])
        P = self.CostFunc.P_fun(self.X_goal, x[-1])

        k = [None] * self.step
        d = [None] * self.step

        Qu_list = [None] * self.step
        Quu_list = [None] * self.step
        for i in reversed(range(self.step)):
            # \frac{\partial f}{\partial  x}
            dfdx = self.VehicleDynamic.dfdx_func(x[i], u[i])
            # \frac{\partial f}{\partial u}
            dfdu = self.VehicleDynamic.dfdu_func(x[i], u[i])

            lx = self.CostFunc.lx_fun(self.ref_state[i], x[i], u[i])
            lu = self.CostFunc.lu_fun(self.ref_state[i], x[i], u[i])
            lxx = self.CostFunc.lxx_fun(self.ref_state[i], x[i], u[i])
            lux = self.CostFunc.lux_fun(self.ref_state[i], x[i], u[i])
            luu = self.CostFunc.luu_fun(self.ref_state[i], x[i], u[i])

            cx = self.ConstraintFunc.CxFunc_(x[i], u[i], self.Mu[i], self.Lambda[i])
            cu = self.ConstraintFunc.CuFunc_(x[i], u[i], self.Mu[i], self.Lambda[i])
            cxx = self.ConstraintFunc.CxxFunc_(x[i], u[i], self.Mu[i], self.Lambda[i])
            cux = self.ConstraintFunc.CuxFunc_(x[i], u[i], self.Mu[i], self.Lambda[i])
            cuu = self.ConstraintFunc.CuuFunc_(x[i], u[i], self.Mu[i], self.Lambda[i])

            Qx = lx + p @ dfdx + cx
            Qu = lu + p @ dfdu + cu
            Qxx = lxx + dfdx.T @ P @ dfdx + cxx
            Qux = lux + dfdu.T @ P @ dfdx + cux
            Quu = luu + dfdu.T @ P @ dfdu + cuu
            Quu_=np.copy(Quu)
            beta=self.beta
            while not self.is_pos_def(Quu_):
                Quu_+=beta*np.eye(U_DIM)
                beta *= 2
            Quu_list[i] = Quu
            Qu_list[i]  = Qu
            b=np.hstack((Qu.full().T,Qux.full()))
            aug_x = - ca.DM(np.linalg.solve(Quu,b))
            k[i]=aug_x[:,1:]
            d[i]=aug_x[:,0]
            #Quu_inverse = self.regularized_persudo_inverse(Quu)
            #k[i]=-Quu_inverse@Qux
            #d[i]=-Quu_inverse@Qu.T
            p = Qx  + d[i].T @ Quu @ k[i] + d[i].T @ Qux + Qu @ k[i]
            P = Qxx + k[i].T @ Quu @ k[i] + Qux.T @ k[i] + k[i].T @ Qux
        return k, d, Qu_list, Quu_list
    def ForWardPass(self,x,u,k,d,alpha,Qu_list,Quu_list):
        '''
        the ForWard process in ILQR
        @param x: the state input
        @param u: the control input
        @param k: the back term
        @param d: the back term
        @param alpha: line search item
        @param Qu_list: the list of Qu
        @param Quu_list: the list of Quu
        @return:
        '''
        x_new = []
        u_new = []
        x_new.append(x[0])
        delta_J = 0.0
        for i in range(self.step):
            u_new.append(u[i] + k[i] @ (x_new[i]-x[i]) + alpha*d[i])
            x_new.append(self.VehicleDynamic.rk4_func(x_new[i],u_new[i]))
            # \delta J += \alpha \times (Q_{u}d+\frac{1}{2}\alpha ^2(d_i^TQ_{uu}d_i))
            delta_J += alpha * ( Qu_list[i] @ d[i]) + 0.5 * pow(alpha,2) * (d[i].T @ Quu_list[i] @ d[i])

        delta_x_terminal = x_new[-1] - x[-1]
        delta_J += (delta_x_terminal.T @ self.CostFunc.P_fun(self.ref_state[-1],x[-1]) @ delta_x_terminal +
                    self.CostFunc.p_fun(self.ref_state[-1],x[-1]) @delta_x_terminal)

        J = self.Evalueate(x_new,u_new)
        return x_new,u_new,J,delta_J

    def Solve(self,u_init):
        '''
        using this function to solve the nolinear planning problem
        @param x_init: the init vehicle state
        @param u_init: the init control input of vehicle
        @return:
        '''
        if self.needPrint:
            print("============== AL-ILQR starts ==============")
        # Init trajectory and control input
        u = u_init
        x = self.init_trajectory(u)

        x_hist = []
        u_hist = []
        ALILQR_iter = 0
        while True:
            if self.needPrint:
                print(
                    "ALILQR: New al-ilqr iteration {0} starts ...".format(ALILQR_iter))
            if ALILQR_iter >= self.max_iter:
                print("ALILQR: Reach ilqr maximum iteration number")
                break
            J_opt = self.Evalueate(x, u)
            # ilqr Main loop
            ilqr_iter = 0
            converged = False
            while not converged:
                if self.needPrint:
                    print(
                        "ALILQR: New ilqr iteration {0} starts ...".format(ilqr_iter))
                if ilqr_iter >= self.max_iter:
                    print("ALILQR: Reach ilqr maximum iteration number")
                    break
                # Backward pass
                K, k, Qu_list, Quu_list = self.BackWardPass(x, u)
                # Line search
                alpha = 1.0
                J_new = 0.0
                accept = False
                while not accept:
                    if alpha < 1e-6:
                        print("ALILQR: Line search fail to decrease cost function")
                        accept=True
                        break
                    # Forward pass
                    x_new, u_new, J_new, delta_J = self.ForWardPass(
                        x, u, K, k, alpha, Qu_list, Quu_list)
                    z = (J_opt - J_new) / -delta_J
                    if self.needPrint:
                        print("ALILQR: J_opt:{0} J_new:{1} delta_J:{2} z:{3}".format(
                            J_opt, J_new, delta_J, z))
                    if ((J_opt - J_new)/J_opt < self.J_tolerance or (z > self.line_search_beta_1 and z < self.line_search_beta_2)) and (J_new<J_opt):
                        x = x_new
                        u = u_new
                        accept = True
                    alpha *= self.line_search_gamma
                if accept:
                    if abs(J_opt - J_new)/J_opt < self.J_tolerance:
                        converged = True
                    J_opt = J_new
                    x_hist.append(x)
                    u_hist.append(u)
                ilqr_iter += 1
            # ConstraintViolation = self.CalcConstraintViolation(x_hist[-1],u_hist[-1])
            ConstraintViolation = self.CalcMaxVio(x_hist[-1],u_hist[-1])
            if self.needPrint:
                print("ALILQR: New al-ilqr iteration {0} ends ... constraint violation: {1}".format(
                    ALILQR_iter, ConstraintViolation))
            if ConstraintViolation < self.ConstraintTolerance:
                break
            self.UpdatePenalityParam(x,u)
            ALILQR_iter += 1

        res_dict = {'x_hist': x_hist, 'u_hist': u_hist}
        if self.needPrint:
            print("============== AL-ILQR ends ==============")
        return res_dict


def visualize_results(x_traj, u_traj, Tf, N):
    """
    可视化优化控制问题的结果，包括静态图表和动画，并显示变量的上下界
    """
    time = np.linspace(T0, Tf, N)

    # 状态变量的上下界
    state_bounds = {
        'v (velocity)': (V_MIN, V_MAX),
        'phi': (-PHI_MAX, PHI_MAX)
        # 其他状态变量如果有上下界，可以在此添加
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
        axs2[i].plot(time, u_traj[:, i], label=control_labels[i], color='orange')
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
    anim.save('vehicle_trajectory.gif', writer='pillow')

def cost_true(x,u):
    cost=0
    for i in range(N):
        cost+=0.5*u[i].T@np.eye(2)@u[i]
    diff_goal=np.linalg.norm(x[-1]-XN.T)
    return cost, diff_goal

def trajectory_generate(Vehicle,Cost,Constraint, x_start,x_goal,u_initial_guess,Nt,needPrint=True):
    solver=ALILQR(Vehicle,Cost,Constraint, x_start,x_goal,Nt,needPrint)
    res = solver.Solve(u_initial_guess)
    x_opt = res['x_hist'][-1]
    u_opt = res['u_hist'][-1]
    State = []
    Control = []
    for i in range(1,len(x_opt)):
        State.append(x_opt[i].full().flatten())
        Control.append(u_opt[i-1].full().flatten())
    State = np.array(State)
    Control = np.array(Control)
    return State, Control

if __name__ == "__main__":
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

    #等式约束
    eq_constraints = []
    #eq_constraints.append(a[-1] - UN[0])
    #eq_constraints.append(omega[-1] - UN[1])
    #eq_constraints.append(x[-1]-XN[0])
    #eq_constraints.append(y[-1] - XN[1])
    #eq_constraints.append(v[-1] - XN[2])
    #eq_constraints.append(phi[-1] - XN[3])
    #eq_constraints.append(theta[-1] - XN[4])
    Q=np.eye(X_DIM)*0.00
    R=np.eye(U_DIM)*1
    Qn=np.eye(X_DIM)*100
    Constraint = ConstraintFunction(eq_constraints, ineq_constraints, state,control)
    Cost = CostFunc(XN, Q, R, Qn,N)
    Vehicle=VehicleDynamic(LW,H)
    np.random.seed(7)
    u_initial_guess = np.random.uniform(low=[A_MIN, -OMEGA_MAX], high=[A_MAX, OMEGA_MAX], size=(N, U_DIM))*0.1
    import  time
    time_start = time.time()
    State,Control=trajectory_generate(Vehicle, Cost, Constraint, X0, XN, u_initial_guess, N)
    true_cost,diff_goal=cost_true(State, Control)
    print('success,using time(s):', time.time() - time_start)
    print('The True Cost is:', true_cost)
    print('The diff in the goal is :', diff_goal)
    visualize_results(State, Control, Tf, N)
    print("Gif graph saved successfully!")