import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры системы
m = 1.0  # масса маятника
l = 1.0  # длина маятника
g = 9.81  # ускорение свободного падения
k = 5.0  # жесткость пружины

# Частоты нормальных мод
omega1 = np.sqrt((2 * k / m) - (g / l))
omega2 = np.sqrt((2 * k / m) + (g / l))

assert omega1 > 0 and omega2 > 0

# Функция для численного моделирования линеаризованной системы
def linearized_system(t, y):
    phi1, phi2, phi1_dot, phi2_dot = y
    dphi1_dt = phi1_dot
    dphi2_dt = phi2_dot
    dphi1_dot_dt = -(g / l) * phi1 - (k / m) * (phi1 - phi2)
    dphi2_dot_dt = -(g / l) * phi2 - (k / m) * (phi2 - phi1)
    return [dphi1_dt, dphi2_dt, dphi1_dot_dt, dphi2_dot_dt]

# Функция для численного моделирования исходной (нелинейной) системы
def nonlinear_system(t, y):
    phi1, phi2, phi1_dot, phi2_dot = y
    dphi1_dt = phi1_dot
    dphi2_dt = phi2_dot
    dphi1_dot_dt = -(g / l) * np.sin(phi1) - (k / m) * (phi1 - phi2)
    dphi2_dot_dt = -(g / l) * np.sin(phi2) - (k / m) * (phi2 - phi1)
    return [dphi1_dt, dphi2_dt, dphi1_dot_dt, dphi2_dot_dt]

# Начальные условия для нормальных мод (режим нормальных колебаний)
A1 = 1.0  # амплитуда первой моды
A2 = 1.0  # амплитуда второй моды
phi1_0 = A1 + A2
phi2_0 = A1 - A2
phi1_dot_0 = omega1 * A1 + omega2 * A2
phi2_dot_0 = omega1 * A1 - omega2 * A2

# Время моделирования
t_span = (0, 10)  # от 0 до 10 секунд
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # точки времени для вывода

# Численное решение для линеаризованной системы
sol_linear = solve_ivp(
    linearized_system, t_span, [phi1_0, phi2_0, phi1_dot_0, phi2_dot_0], t_eval=t_eval
)

# Численное решение для исходной системы
sol_nonlinear = solve_ivp(
    nonlinear_system, t_span, [phi1_0, phi2_0, phi1_dot_0, phi2_dot_0], t_eval=t_eval
)

# Построение графиков
plt.figure(figsize=(12, 6))
plt.plot(sol_linear.t, sol_linear.y[0], label="Линеаризованная система: $\\phi_1$")
plt.plot(sol_linear.t, sol_linear.y[1], label="Линеаризованная система: $\\phi_2$")
plt.plot(sol_nonlinear.t, sol_nonlinear.y[0], "--", label="Исходная система: $\\phi_1$")
plt.plot(sol_nonlinear.t, sol_nonlinear.y[1], "--", label="Исходная система: $\\phi_2$")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Сравнение линеаризованной и исходной систем в режиме нормальных колебаний")
plt.legend()
plt.grid()
plt.show()
