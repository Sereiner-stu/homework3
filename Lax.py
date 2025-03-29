import numpy as np
import matplotlib.pyplot as plt

#参数设置
L = 3.0 # 空间域长度
T = 3.0 # 总时间
Nx = 300 # 空间网格数
a = 1 # 波动方程系数
c = 0.9 # c = a * dt / dx

#参数计算
dx = L / Nx # 空间步长
dt = c * dx / a # 时间步长
Nt = int(T / dt) # 时间网格数
x = np.linspace(0, L, Nx + 1) # 0 到 L 上的空间网格坐标

u_new = np.sin(2 * np.pi * x)
u_exact = u_new.copy()

for n in range(Nt):
    u_old = u_new.copy()

    u_new[1 : -1] = 0.5 * (1 - c) * u_old[2 : ] + 0.5 * (1 + c) * u_old[ : -2] # Lax格式迭代

    u_new[0] = 0.5 * (1 - c) * u_old[1] + 0.5 * (1 + c) * u_old[-2] #边界条件
    u_new[-1] = u_new[0] # 周期条件

    t = (n + 1) * dt
    u_exact = np.sin(2 * np.pi * (x - a * t))

plt.figure(figsize=(10,6))
plt.plot(x, u_new, 'r--', lw=2, label='Lax格式数值解')
plt.plot(x, u_exact, 'k-', lw=1, alpha=0.8, label='精确解')
plt.title(f"一阶波动方程解比较 ")
plt.xlabel('空间位置 x')
plt.ylabel('幅值 u')
plt.legend()
plt.grid(True)
plt.show()