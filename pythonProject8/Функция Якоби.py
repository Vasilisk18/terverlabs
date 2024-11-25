import math
import numpy as np
import matplotlib.pyplot as plt

# Определение параметров системы
C = 1e-9  # Фарад
C1 = 1e-8  # Фарад
C2 = 1e-7  # Фарад
L1 = 0.25  # Генри

# Расчёт частоты первой нормальной моды
omega1 = 1 / L1 * (1 / C1 + 1 / C)

# Функция для вычисления частоты второй нормальной моды
def omega2(L2):
    return 1 / L2 * (1 / C2 + 1 / C)

# Установим значение L2
L2 = 0.115
om1 = math.sqrt(omega1)
om2 = math.sqrt(omega2(L2))

# Коэффициенты временных зависимостей
T1 = -C * om2 / (2 * om1 * (1 - om2 / om1))
T2 = C * C1 / (4 * (C + C1) * (1 - om2 / om1))

# Функции для описания колебаний
def y1(t):
    return math.cos(om1 * t) + 2 * (C + C1) / C1 * math.cos(om2 * t)

def y2(t):
    return math.cos(om2 * t) - 2 * (C + C1) / C1 * math.cos(om1 * t)

# Создание временной шкалы
T = np.arange(0, 0.2, 0.0005)

# Вычисление временных реализаций
u1 = [y1(t) for t in T]
u2 = [y2(t) for t in T]

# Построение графика для биений
plt.figure(figsize=(12, 7))
plt.plot(T, u1, "r-", label="y1(t)")
plt.plot(T, u2, "b-", label="y2(t)")
plt.xlabel("Время (с)")
plt.ylabel("Амплитуда")
plt.title("Режим биений")
plt.legend()
plt.grid()
plt.show()

# Новые параметры для второй части кода
C = 1e-8  # Фарад
C1 = 1e-7  # Фарад
C2 = 1e-6  # Фарад
L1 = 0.125  # Генри

# Перерасчёт частоты первой нормальной моды
omega1 = 1 / L1 * (1 / C1 + 1 / C)

# Функция для вычисления частоты второй нормальной моды
def omega2(L2):
    return 1 / L2 * (1 / C2 + 1 / C)

# Функции для анализа нормальных мод
def x1(t, alpha):
    return 1 / math.sqrt(2) * (
        math.sqrt(
            omega1 + omega2(t) - math.sqrt((omega1 - omega2(t))**2 + 4 * alpha**2)
        )
    )

def x2(t, alpha):
    return 1 / math.sqrt(2) * (
        math.sqrt(
            omega1 + omega2(t) + math.sqrt((omega1 - omega2(t))**2 + 4 * alpha**2)
        )
    )

# Параметр альфа и временная шкала
alpha = 0.01
x = np.arange(0.001, 0.5, 0.001)

# Вычисление зависимостей
y1 = [x1(xi, alpha) for xi in x]
y2 = [x2(xi, alpha) for xi in x]

# Построение графиков для нормальных мод
plt.figure(figsize=(10, 10))
plt.xlim([0.11, 0.12])
plt.ylim([27500, 32500])
plt.plot(x, y1, "-r", label="x1(t)")
plt.plot(x, y2, "-b", label="x2(t)")
plt.xlabel("Параметр")
plt.ylabel("Частота")
plt.title("Нормальные моды")
plt.legend()
plt.grid()
plt.show()
