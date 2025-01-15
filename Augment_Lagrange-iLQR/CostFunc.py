import casadi as ca
import numpy as np


class CostFunc:
    def __init__(self, Xn, Q , R, Q_Terminal,Nt,ref_x=None):
        self.ref_path = np.ones((Nt,np.size(Q_Terminal,0)))*Xn if ref_x is None else ref_x
        self.Q = Q
        self.R = R
        self.Q_Terminal = Q_Terminal

        self.StateCostFunc()
        self.CalcJacobian()
        self.CalcHessian()
        self.CalcTerminalFunc()
    def StateCostFunc(self):
        '''
        use to generate the State Cost Function and it's Terminal Cost Function
        @return:
        '''
        x_ref = ca.SX.sym('x_ref')
        y_ref = ca.SX.sym('y_ref')
        phi_ref = ca.SX.sym('phi_ref')
        v_ref = ca.SX.sym('v_ref')
        theta_ref= ca.SX.sym('theta_ref')
        self.state_ref = ca.vertcat(x_ref, y_ref, v_ref,phi_ref, theta_ref)
        x, y, v, phi, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('v'), ca.SX.sym('phi'), ca.SX.sym('theta')
        a, omega = ca.SX.sym('a'), ca.SX.sym('omega')
        self.state = ca.vertcat(x, y, v, phi, theta)
        self.control= ca.vertcat(a, omega)
        state_diff = self.state - self.state_ref

        self.StageCost = state_diff.T @ self.Q @ state_diff
        self.StageCost += self.control.T @ self.R @ self.control

        self.TerminalCost = state_diff.T @ self.Q_Terminal @ state_diff
        #self.TerminalCost += self.control.T @ self.R_Terminal @ self.control

        self.StageCostFunc = ca.Function("StageCost",[self.state_ref,self.state,self.control],[self.StageCost])
        self.TerminalCostFunc = ca.Function("TerminalCost",[self.state_ref,self.state,self.control],[self.TerminalCost])
    def CalcCost(self,State,Control):
        '''
        function : use to calculate the trajectory Cost
        @param State: the vehicle state ,the format is [x ,y ,yaw ,velocity ]
        @param Control:the control input of vehicle the format is [steering,a]
        @return:
        '''
        Cost = 0
        self.StageCostFunction = 0
        for i in range(len(self.ref_path) - 1):
            Cost += self.StageCostFunc(self.ref_path[i], State[i], Control[i])
            self.StageCostFunction += self.StageCostFunc(self.ref_path[i], self.state, self.control)
            # print(Cost,i)
        Cost += self.TerminalCostFunc(self.ref_path[-1], State[-1],Control[-1])
        # print(Cost)
        return Cost
    def CalcJacobian(self):
        '''
        Calculate the Jacobian of Cost Function
        '''
        self.lx = ca.jacobian(self.StageCost,self.state)
        self.lu = ca.jacobian(self.StageCost,self.control)
        self.lx_fun = ca.Function("lx",[self.state_ref,self.state,self.control],[self.lx])
        self.lu_fun = ca.Function("lu",[self.state_ref,self.state,self.control],[self.lu])
    def CalcHessian(self):
        '''
        Calculate the Hessian of Cost Function
        '''
        self.lxx_fun = ca.Function("lxx",[self.state_ref,self.state,self.control],[ca.jacobian(self.lx,self.state)])
        self.lux_fun = ca.Function("lux",[self.state_ref,self.state,self.control],[ca.jacobian(self.lu,self.state)])
        self.luu_fun = ca.Function("luu",[self.state_ref,self.state,self.control],[ca.jacobian(self.lu,self.control)])

    def CalcTerminalFunc(self):
        '''
        Calculate the Jacobian and Hessian of Terminal Cost Function
        '''
        p = ca.jacobian(self.TerminalCost,self.state)
        self.p_fun = ca.Function("p",[self.state_ref,self.state],[p])
        self.P_fun = ca.Function("P",[self.state_ref,self.state],[ca.jacobian(p,self.state)])






class CostFunc_tacking:
    def __init__(self, Xn, Q , R, Q_Terminal,Nt,ref_x=None,ref_u=None):
        self.ref_path = np.ones((Nt,np.size(Q_Terminal,0)))*Xn if ref_x is None else ref_x
        self.ref_u=ref_u
        self.Q = Q
        self.R = R
        self.Q_Terminal = Q_Terminal

        self.StateCostFunc()
        self.CalcJacobian()
        self.CalcHessian()
        self.CalcTerminalFunc()
    def StateCostFunc(self):
        '''
        use to generate the State Cost Function and it's Terminal Cost Function
        @return:
        '''
        x_ref = ca.SX.sym('x_ref')
        y_ref = ca.SX.sym('y_ref')
        phi_ref = ca.SX.sym('phi_ref')
        v_ref = ca.SX.sym('v_ref')
        theta_ref= ca.SX.sym('theta_ref')
        a_ref= ca.SX.sym('a_ref')
        omega_ref= ca.SX.sym('omega_ref')
        self.state_ref = ca.vertcat(x_ref, y_ref, v_ref,phi_ref, theta_ref)
        x, y, v, phi, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('v'), ca.SX.sym('phi'), ca.SX.sym('theta')
        self.control_ref=ca.vertcat(a_ref,omega_ref)
        self.state = ca.vertcat(x, y, v, phi, theta)
        a, omega = ca.SX.sym('a'), ca.SX.sym('omega')
        self.control= ca.vertcat(a, omega)
        state_diff = self.state - self.state_ref
        control_diff = self.control-self.control_ref

        self.StageCost = state_diff.T @ self.Q @ state_diff
        self.StageCost += control_diff.T @ self.R @ control_diff

        self.TerminalCost = state_diff.T @ self.Q_Terminal @ state_diff
        #self.TerminalCost += self.control.T @ self.R_Terminal @ self.control

        self.StageCostFunc = ca.Function("StageCost",[self.state_ref,self.control_ref,self.state,self.control],[self.StageCost])
        self.TerminalCostFunc = ca.Function("TerminalCost",[self.state_ref,self.state,self.control],[self.TerminalCost])
    def CalcCost(self,State,Control):
        '''
        function : use to calculate the trajectory Cost
        @param State: the vehicle state ,the format is [x ,y ,yaw ,velocity ]
        @param Control:the control input of vehicle the format is [steering,a]
        @return:
        '''
        Cost = 0
        self.StageCostFunction = 0
        for i in range(len(self.ref_path) - 1):
            Cost += self.StageCostFunc(self.ref_path[i],self.ref_u[i], State[i], Control[i])
            self.StageCostFunction += self.StageCostFunc(self.ref_path[i],self.ref_u[i], self.state, self.control)
            # print(Cost,i)
        Cost += self.TerminalCostFunc(self.ref_path[-1], State[-1],Control[-1])
        # print(Cost)
        return Cost
    def CalcJacobian(self):
        '''
        Calculate the Jacobian of Cost Function
        '''
        self.lx = ca.jacobian(self.StageCost,self.state)
        self.lu = ca.jacobian(self.StageCost,self.control)
        self.lx_fun = ca.Function("lx",[self.state_ref,self.control_ref,self.state,self.control],[self.lx])
        self.lu_fun = ca.Function("lu",[self.state_ref,self.control_ref,self.state,self.control],[self.lu])
    def CalcHessian(self):
        '''
        Calculate the Hessian of Cost Function
        '''
        self.lxx_fun = ca.Function("lxx",[self.state_ref,self.control_ref,self.state,self.control],[ca.jacobian(self.lx,self.state)])
        self.lux_fun = ca.Function("lux",[self.state_ref,self.control_ref,self.state,self.control],[ca.jacobian(self.lu,self.state)])
        self.luu_fun = ca.Function("luu",[self.state_ref,self.control_ref,self.state,self.control],[ca.jacobian(self.lu,self.control)])

    def CalcTerminalFunc(self):
        '''
        Calculate the Jacobian and Hessian of Terminal Cost Function
        '''
        p = ca.jacobian(self.TerminalCost,self.state)
        self.p_fun = ca.Function("p",[self.state_ref,self.state],[p])
        self.P_fun = ca.Function("P",[self.state_ref,self.state],[ca.jacobian(p,self.state)])