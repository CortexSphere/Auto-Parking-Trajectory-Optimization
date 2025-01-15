import numpy as np
import casadi as ca

class VehicleDynamicNoised:
    def __init__(self, lw, dt, noise_scale=0,process_noise_std=np.zeros(5)):
        self.LW = lw
        self.dt = dt  # 积分步长
        self.process_noise_std = process_noise_std  # 各状态变量的过程噪声标准差
        self.noise_scale = noise_scale
        # 定义符号变量
        x, y, v, phi, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('v'), ca.SX.sym('phi'), ca.SX.sym('theta')
        a, omega = ca.SX.sym('a'), ca.SX.sym('omega')
        self.state = ca.vertcat(x, y, v, phi, theta)
        self.control = ca.vertcat(a, omega)

        # 定义动态方程
        self.dx = ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            a,
            omega,
            v * ca.tan(phi) / self.LW
        )
        self.Dynamic_Func = ca.Function('f', [self.state, self.control], [self.dx])

        # 构建RK4积分器
        self.RK4()
        self.Jacobian()
        self.Hessian()

    def RK4(self):
        x = self.state
        u = self.control
        dt = self.dt
        k1 = self.Dynamic_Func(x, u)
        k2 = self.Dynamic_Func(x + 0.5 * dt * k1, u)
        k3 = self.Dynamic_Func(x + 0.5 * dt * k2, u)
        k4 = self.Dynamic_Func(x + dt * k3, u)
        self.rk4 = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.rk4_func = ca.Function('RK4', [self.state, self.control], [self.rk4])

    def propagate(self, x, u):
        x_casadi = ca.DM(x)
        u_casadi = ca.DM(u)
        x_next = self.rk4_func(x_casadi, u_casadi).full().flatten()

        # 添加过程噪声
        noise=self.noise_scale * x_next * np.random.normal(0, self.process_noise_std, size=x_next.shape)
        x_next += noise

        return x_next

    def Jacobian(self):
        self.dfdx = ca.jacobian(self.rk4, self.state)
        self.dfdu = ca.jacobian(self.rk4, self.control)
        self.dfdx_func = ca.Function('dfdx', [self.state, self.control], [self.dfdx])
        self.dfdu_func = ca.Function('dfdu', [self.state, self.control], [self.dfdu])

    def Hessian(self):
        self.dfdxdx = ca.jacobian(self.dfdx, self.state)
        self.dfdudu = ca.jacobian(self.dfdu, self.control)
        self.dfdxdu = ca.jacobian(self.dfdx, self.control)
        self.dfdudx = ca.jacobian(self.dfdu, self.state)
        self.dfdxdx_func = ca.Function('dfdxdx', [self.state, self.control], [self.dfdxdx])
        self.dfdudu_func = ca.Function('dfdudu', [self.state, self.control], [self.dfdudu])
        self.dfdxdu_func = ca.Function('dfdxdu', [self.state, self.control], [self.dfdxdu])
        self.dfdudx_func = ca.Function('dfdudx', [self.state, self.control], [self.dfdudx])
