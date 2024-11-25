import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Определение нелинейной системы
def rhs(t, X):
    x, y = X
    dxdt = y
    dydt = x ** 4 - 5 * x ** 2 + 4
    return [dxdt, dydt]


# Вычисление матрицы Якоби в заданной точке
def jacobian_matrix(x):
    return np.array([[0, 1],
                     [4 * x ** 3 - 10 * x, 0]])


# Вычисление собственных значений матрицы Якоби
def compute_eigen(x):
    J = jacobian_matrix(x)
    eig_vals, eig_vecs = np.linalg.eig(J)
    return eig_vals, eig_vecs

# Форматирование собственных чисел для отображения
def format_eigen(eig):
    if np.iscomplex(eig):
        sign = '+' if eig.imag >= 0 else '-'
        return f"{eig.real:.2f} {sign} {abs(eig.imag):.2f}j"
    else:
        return f"{eig:.2f}"


# Поиск равновесных точек
def find_equilibrium_points():
    # Решаем x^4 -5x^2 +4=0
    coeffs = [1, 0, -5, 0, 4]
    roots = np.roots(coeffs)
    # Оставляем только вещественные корни
    equilibria = np.real(roots[np.isreal(roots)])
    return equilibria

def seps(x, y, eig_vecs, scale=0.0001):
    # Берем первые и вторые собственные векторы
    vec1, vec2 = eig_vecs[:, 0], eig_vecs[:, 1]

    # Начальные точки вдоль и против каждого собственного вектора
    x1, y1 = x + vec1[0] * scale, y + vec1[1] * scale
    x2, y2 = x + vec2[0] * scale, y + vec2[1] * scale
    x12, y12 = x - vec1[0] * scale, y - vec1[1] * scale
    x22, y22 = x - vec2[0] * scale, y - vec2[1] * scale

    # Возвращаем начальные точки для интеграции
    return [(x1, y1), (x2, y2), (x12, y12), (x22, y22)]

def plot_separatrices(ax, eq_x, eq_y=0):
    eig_vals, eig_vecs = compute_eigen(eq_x)

    # Проверяем, является ли точка седлом
    is_saddle = np.any(eig_vals.real > 0) and np.any(eig_vals.real < 0)
    if not is_saddle:
        return

    # Получаем начальные точки вдоль собственных векторов
    points = seps(eq_x, eq_y, eig_vecs, scale=0.1)

    # Интегрируем траектории от каждой начальной точки
    for (x0, y0) in points:
        # Интеграция вперед
        sol = solve_ivp(rhs, [0, 25], [x0, y0], t_eval=np.linspace(0, 25, 500), max_step=0.05)
        ax.plot(sol.y[0], sol.y[1], color='yellow', linestyle='--', linewidth=1.5)

        # Интеграция назад
        sol_back = solve_ivp(rhs, [0, -25], [x0, y0], t_eval=np.linspace(0, -25, 500), max_step=0.05)
        ax.plot(sol_back.y[0], sol_back.y[1], color='yellow', linestyle='--', linewidth=1.5)

# Функция для построения фазового портрета с использованием обновлённых сепаратрис
def plot_phase_portrait():
    fig, ax = plt.subplots(figsize=(15, 10))

    # Настройка границ
    x_min, x_max = -4, 4
    y_min, y_max = -4, 4
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')
    ax.set_ylabel("y = x'")
    ax.set_title("Фазовый портрет системы")

    # Создание направляющего поля
    N = 25  # Разрешение сетки
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x, y)
    U_field = Y
    V_field = X ** 4 - 5 * X ** 2 + 4
    M = np.hypot(U_field, V_field)
    M[M == 0] = 1
    U_field /= M
    V_field /= M
    ax.quiver(X, Y, U_field, V_field, M, pivot='mid', cmap='gray', alpha=0.8)

    # Поиск и отображение равновесных точек
    equilibria = find_equilibrium_points()
    for eq_x in equilibria:
        eq_y = 0
        eig_vals, eig_vecs = compute_eigen(eq_x)

        # Проверка на тип равновесной точки
        if eq_x == 2 or eq_x == -1:
            ax.plot(eq_x, eq_y, 'x', color='red', markersize=10)
        # Проверка на тип равновесной точки (центр)
        else:  # Устойчивая точка
            ax.plot(eq_x, eq_y, 'o', color='blue', markersize=10)
            circle = plt.Circle((eq_x, eq_y), 0.3, color='green', fill=False, linestyle='-')
            ax.add_artist(circle)
            circle = plt.Circle((eq_x, eq_y), 0.2, color='green', fill=False, linestyle='-')
            ax.add_artist(circle)

        # Рисование сепаратрис для седловых точек
        plot_separatrices(ax, eq_x, eq_y)

    # Добавление траекторий из различных начальных условий
    initial_conditions = [
        (2.5, 0), (-2.5, 0),
        (0, 2), (0, -2),
        (1.5, 0), (-1.5, 0),
        (2, 2), (-2, -2),
        (1, 1), (-1, -1)
    ]
    t_span = [0, 25]
    for ic in initial_conditions:
        sol_back = solve_ivp(rhs, [t_span[1], 0], ic, dense_output=True, max_step=0.05)
        ax.plot(sol_back.y[0], sol_back.y[1], color='red', linewidth=0.8, alpha=0.8)
        sol_back = solve_ivp(rhs, [-t_span[1], 0], ic, dense_output=True, max_step=0.05)
        ax.plot(sol_back.y[0], sol_back.y[1], color='red', linewidth=0.8, alpha=0.8)

    plt.grid(True)
    plt.show()


def main():
    plot_phase_portrait()


if __name__ == "__main__":
    main()
