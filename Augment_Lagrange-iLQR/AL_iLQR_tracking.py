from Vehicle_Dynamic_noised import VehicleDynamicNoised

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




class ALILQR_:
    def __init__(self,VehicleDynamic:VehicleDynamicNoised,CostFunc:CostFunc,ConstraintFunc:ConstraintFunction, X_start,X_goal,Nt,ref_X=None,ref_U=None,needPrint=True):
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
        self.ref_state = np.ones((Nt+1,X_DIM))*X_goal if ref_X is None else ref_X
        if ref_U is not None:
            self.ref_control = ref_U
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

            lx = self.CostFunc.lx_fun(self.ref_state[i],self.ref_control[i], x[i], u[i])
            lu = self.CostFunc.lu_fun(self.ref_state[i],self.ref_control[i], x[i], u[i])
            lxx = self.CostFunc.lxx_fun(self.ref_state[i],self.ref_control[i], x[i], u[i])
            lux = self.CostFunc.lux_fun(self.ref_state[i],self.ref_control[i], x[i], u[i])
            luu = self.CostFunc.luu_fun(self.ref_state[i],self.ref_control[i], x[i], u[i])

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




def trajectory_tracking(Vehicle,Cost,Constraint, x_start,x_goal,u_initial_guess,Nt,ref_X,ref_U,needPrint=True):
    solver=ALILQR_(Vehicle,Cost,Constraint, x_start,x_goal,Nt,ref_X,ref_U,needPrint)
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

