import numpy as np
import matplotlib.pyplot as plt

#参数设置
L = 3.0 # 空间域长度
T = 1.0 # 总时间
Nx = 300 # 空间网格数
a = 1 # 波动方程系数
c = 0.8 # c = a * dt / dx

#参数计算
dx = L / Nx # 空间步长
dt = c * dx / a # 时间步长
Nt = int(T / dt) # 时间网格数
x = np.linspace(0, L, Nx + 1) # 0 到 L 上的空间网格坐标

u_new = np.sin(2 * np.pi * x)
u_exact = u_new.copy()

for n in range(Nt):
    u_old = u_new.copy()

    u_new[1 : -1] = u_old[1 : -1] - c * (u_old[1 : -1] - u_old[ : -2]) # 一阶迎风格式

    u_new[0] = u_old[0] - c * (u_old[0] - u_old[-2]) # 边界条件
    u_new[-1] = u_new[0] # 周期条件

    t = (n + 1) * dt
    u_exact = np.sin(2 * np.pi * (x - a * t))

plt.figure(figsize=(10,6))
plt.plot(x, u_new, 'r--', lw=2, label='First order upwind format numerical solution')
plt.plot(x, u_exact, 'k-', lw=1, alpha=0.8, label='exact solution')
plt.title(f"Comparison of First Order Wave Equation Solutions")
plt.xlabel('spatial position x')
plt.ylabel('amplitude u')
plt.legend()
plt.grid(True)
plt.show()