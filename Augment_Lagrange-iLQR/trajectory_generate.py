from AL_iLQR import *
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