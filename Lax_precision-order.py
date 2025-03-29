import numpy as np
import matplotlib.pyplot as plt


def compute_error(Nx, T = 1.0, a = 1.0, c = 0.8):
    """L2误差"""
    # 参数设置
    L = 3.0 # 空间域长度

    # 参数计算
    dx = L / Nx # 空间步长
    dt = c * dx / a # 时间步长
    Nt = int(T / dt) # 时间网格数
    x = np.linspace(0, L, Nx + 1) # 0 到 L 上的空间网格坐标

    u_new = np.sin(2 * np.pi * x)
    u_exact = u_new.copy()

    for n in range(Nt):
        u_old = u_new.copy()

        u_new[1: -1] = 0.5 * (1 - c) * u_old[2:] + 0.5 * (1 + c) * u_old[: -2] # Lax格式迭代

        u_new[0] = 0.5 * (1 - c) * u_old[1] + 0.5 * (1 + c) * u_old[-2] # 边界条件
        u_new[-1] = u_new[0] # 周期条件

        t = (n + 1) * dt
        u_exact = np.sin(2 * np.pi * (x - a * t))

    # 计算L2误差
    error = np.linalg.norm(u_new - u_exact, 2) * np.sqrt(dx) # 将离散L2范数转换为连续L2误差的近似
    return error


# 依次加密网格数
Nx_list = [50, 100, 200, 400, 800]
errors = []
for Nx in Nx_list:
    error = compute_error(Nx)
    errors.append(error)
    print(f"Nx={Nx}, dx={3 / Nx:.6f}, error={error:.6e}")


# 绘制误差与dx关系图
dx_list = [3.0 / Nx for Nx in Nx_list]
plt.figure(figsize=(8, 5))
plt.loglog(dx_list, errors, 'o-', label='L2 error')
plt.loglog(dx_list, [errors[0] * (dx / dx_list[0]) for dx in dx_list], 'k--', label='first order slope for reference') # 绘制基于error[0]的一阶收敛的理论参考斜线
plt.loglog(dx_list, [errors[0] * (dx / dx_list[0]) * (dx / dx_list[0]) for dx in dx_list], 'g-.', label='second order slope for reference') # 绘制基于error[0]的二阶收敛的理论参考斜线
plt.xlabel('spatial step size dx (logarithmic coordinates)')
plt.ylabel('L2 error (logarithmic coordinates)')
plt.title('Lax Scheme Precision Order Verification')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()