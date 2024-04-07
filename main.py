import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 
from scipy.linalg import eigh 
 
# Параметры маятников и пружины 
m = 1.0  # масса маятников 
L = 1.0  # длина подвеса 
g = 9.81  # ускорение свободного падения 
k = 100.0  # коэффициент жесткости пружины 
L1 = 0.5  # расстояние до точки крепления пружины 
beta = 0.1  # коэффициент затухания 
 
# Начальные условия: phi1, dphi1/dt, phi2, dphi2/dt 
initial = [0, 0, -3, 0]  # начальные углы и скорости 
 
# Функция правой части системы уравнений 
def dpendulum_dt(Y, t, m, L, g, k, L1, beta): 
    phi1, omega1, phi2, omega2 = Y 
     
    # Система уравнений 
    dphi1_dt = omega1 
    dphi2_dt = omega2 
    domega1_dt = -(g/L)*phi1 - (beta/(m*L))*omega1 + (k*L1/(m*L**2))*(phi2 - phi1) 
    domega2_dt = -(g/L)*phi2 - (beta/(m*L))*omega2 + (k*L1/(m*L**2))*(phi1 - phi2) 
     
    return [dphi1_dt, domega1_dt, dphi2_dt, domega2_dt] 
 
# временной интервал интегрирования (для зависимость от времени) 
t = np.linspace(0, 70, 600) 
 
# решение системы уравнений 
solution = odeint(dpendulum_dt, initial, t, args=(m, L, g, k, L1, beta)) 
 
# разделение решения на компоненты 
phi1, omega1, phi2, omega2 = solution.T 
 
# построение графиков 
plt.figure(figsize=(14, 7)) 
 
# график углов как функция времени 
plt.subplot(2, 1, 1) 
plt.plot(t, phi1, label='phi1(t)') 
plt.plot(t, phi2, label='phi2(t)') 
plt.title('Зависимость углов от времени') 
plt.xlabel('Время (с)') 
plt.ylabel('Угол (рад)') 
plt.legend() 
plt.grid() 
 
# гррафик скоростей как функция времени 
plt.subplot(2, 1, 2) 
plt.plot(t, omega1, label='dphi1/dt(t)') 
plt.plot(t, omega2, label='dphi2/dt(t)') 
plt.title('Зависимость скоростей от времени') 
plt.xlabel('Время (с)') 
plt.ylabel('Скорость угловая (рад/с)') 
plt.legend() 
plt.grid() 
 
# отображение графиков 
plt.tight_layout() 
plt.show() 
 
plt.figure(figsize=(14, 7)) 
 
# график углов как функция времени 
plt.subplot(2, 1, 1) 
plt.plot(t, phi1, label='phi1(t)') 
plt.plot(t, phi2, label='phi2(t)') 
plt.title('Зависимость углов от времени') 
plt.xlabel('Время (с)') 
plt.ylabel('Угол (рад)') 
plt.legend() 
plt.grid() 
 
plt.subplot(2, 1, 2) 
plt.plot(t, omega1, label='dphi1/dt(t)') 
plt.plot(t, omega2, label='dphi2/dt(t)') 
plt.title('Зависимость скоростей от времени') 
plt.xlabel('Время (с)') 
plt.ylabel('Скорость угловая (рад/с)') 
plt.legend() 
plt.grid() 
 
plt.tight_layout() 
 
filename = "pendulum_motion.png" 
 
plt.savefig(filename) 
 
# Матрица жесткости системы, K 
K = np.array([[k*L1 + m*g*L, -k*L1], 
              [-k*L1, k*L1 + m*g*L]]) 
 
# Матрица массы системы, M 
M = np.array([[m*L**2, 0], 
              [0, m*L**2]]) 
 
# Находим собственные значения (квадраты собственных частот) и собственные вектора 
eigenvals, _ = eigh(K, M) 
 
# Нормальные (собственные) частоты являются корнями из собственных значений 
normal_freqs = np.sqrt(np.abs(eigenvals)) 
 
print("Нормальные частоты: ", normal_freqs)