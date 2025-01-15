from IPOPT import *

if __name__ == "__main__":
    # 构造初始值
    x0_guess = 0.01 * np.ones((N + 1) * (U_DIM + X_DIM))  # 初始猜测值
    import time
    time_start = time.time()
    # 求解最优控制问题
    xks, uks = trajectory_generate(x0_guess)
    print('success, using time(s):', time.time() - time_start)
    true_cost, diff_goal = cost_true(xks, uks)
    print('The True Cost is:', true_cost)
    print('The diff in the goal is :', diff_goal)
    visualize_results(xks, uks, Tf, N)
    print("Gif graph saved successfully!")
