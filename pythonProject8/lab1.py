from os import name
from pickle import TRUE
from tkinter.ttk import Scale
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# In[9]:


def f1(gamma):
    def rhs(t, X):
        x, y = X
        return [y, -gamma * y + x ** 4 - 5 * (x ** 2) + 4]

    return rhs

rhs = f1(0.)

# rhs(10., [1.,2.])

def eq_quiver(rhs, limits, N=16):
    xlims, ylims = limits
    xs = np.linspace(xlims[0], xlims[1], N)
    ys = np.linspace(ylims[0], ylims[1], N)
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            vfield = rhs(0.0, [x, y])
            u, v = vfield
            U[i][j] = u
            V[i][j] = v
    return xs, ys, U, V


# In[14]:


def plotonPlane(rhs, limits):
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlabel('x')
    ax.set_ylabel('x\u0307')

    xlims, ylims = limits
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    # xs, ys, U, V = eq_quiver(rhs, limits, 10)
    # plt.quiver(xs, ys, U, V, alpha=0.5, scale = 500)


def plotonPlaneWithoutV(rhs, limits):
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlabel('t')
    ax.set_ylabel('x')

    xlims, ylims = limits
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])


def drawCircle(x, y):
    angle = np.linspace(0, 2 * np.pi, 150)

    radius = 0.05

    x_ = radius * np.cos(angle) + x
    y_ = radius * np.sin(angle) * 2.3 + y

    plt.plot(x_, y_, 'b-')


def drawMark(x, y):
    x1, y1 = [-0.05 + x, 0.05 + x], [y, y]
    plt.plot(x1, y1, 'r-')
    x1, y1 = [x, x], [-0.1 + y, 0.1 + y]
    plt.plot(x1, y1, 'r-')


def drawSep(x, y):
    sol1 = solve_ivp(rhs, [0., 100.], (x, y), method='RK45', rtol=1e-12)
    x1, y1 = sol1.y
    plt.plot(x1, y1, 'y-')

    sol1 = solve_ivp(rhs, [0., -100.], (x, y), method='RK45', rtol=1e-12)
    x1, y1 = sol1.y
    plt.plot(x1, y1, 'y-')


def drawInf(x, y):
    sol1 = solve_ivp(rhs, [0., 100.], (x, y), method='RK45', rtol=1e-12)
    x1, y1 = sol1.y
    plt.plot(x1, y1, 'r-')

    sol1 = solve_ivp(rhs, [0., -100.], (x, y), method='RK45', rtol=1e-12)
    x1, y1 = sol1.y
    plt.plot(x1, y1, 'r-')


def drawUst(x, y):
    sol1 = solve_ivp(rhs, [0., 100.], (x, y), method='RK45', rtol=1e-12)
    x1, y1 = sol1.y
    plt.plot(x1, y1, 'g-')

    sol1 = solve_ivp(rhs, [0., -100.], (x, y), method='RK45', rtol=1e-12)
    x1, y1 = sol1.y
    plt.plot(x1, y1, 'g-')


def drawUZ(x, y):
    sol1 = solve_ivp(rhs, [0., 100.], (x, y), method='RK45', rtol=1e-12)
    x1, y1 = sol1.y
    plt.plot(x1, y1, 'g-')

    sol1 = solve_ivp(rhs, [0., -100.], (x, y), method='RK45', rtol=1e-12)
    x1, y1 = sol1.y
    plt.plot(x1, y1, 'g-')


def drawXT(x, y):
    sol1 = solve_ivp(rhs, [0., 30.], (x, y), method='RK45', rtol=1e-12)
    x2, y2 = sol1.t, sol1.y[0]
    plt.plot(x2, y2, 'r-')

    sol1 = solve_ivp(rhs, [0., -30.], (x, y), method='RK45', rtol=1e-12)
    x2, y2 = sol1.t, sol1.y[0]
    plt.plot(x2, y2, 'r-')


E = 1e-7

draw_dop = True

# gamma = 2 Фазовый портрет системы, при a=2
gamma = 2.
rhs = f1(gamma)
plotonPlane(rhs, [(-3., 3.), (-5., 5.)])
plt.title("Фазовый портрет системы при a = 2")


# -1 0
drawSep(-1. - 1 * E, 0. - 2 * E)
drawSep(-1. + 1 * E, 0. + 2 * E)
drawSep(-1. - 1 * E, 0. + 3 * E)
drawSep(-1. + 1 * E, 0. - 3 * E)

# 2 0
drawSep(2. - 1 * E, 0. + 4 * E)
drawSep(2. + 1 * E, 0. - 4 * E)
drawSep(2. + 1 * E, 0. + 3 * E)
drawSep(2. - 1 * E, 0. - 3 * E)

# infinity tr

drawInf(2., 2.)
drawInf(2.5, 0.)

drawCircle(-2, 0)
drawCircle(1, 0)
drawMark(-1, 0)
drawMark(2, 0)

if draw_dop:
    drawUst(0, 2)
    drawUst(0, 0)
    drawUst(-2, -2)

plt.show()

# gamma = 2√6 Фазовый портрет системы при a=2√6
gamma = 2. * (6. ** 0.5)
rhs = f1(gamma)
plotonPlane(rhs, [(-3., 3.), (-5., 5.)])
plt.title("Фазовый портрет системы при a = 2√6")

# -1 0
drawSep(-1. - 1 * E, 0. + 6 * E)
drawSep(-1. + 1 * E, 0. - 6 * E)
drawSep(-1. + 1 * E, 0. + 1 * E)
drawSep(-1. - 1 * E, 0. - 1 * E)

# 2 0
drawSep(2. - 1 * E, 0. + (- 73 ** 0.5 / 2 + 2.5) * E)
drawSep(2. + 1 * E, 0. - (- 73 ** 0.5 / 2 + 2.5) * E)
drawSep(2. - 1 * E, 0. + (73 ** 0.5 / 2 + 2.5) * E)
drawSep(2. + 1 * E, 0. - (73 ** 0.5 / 2 + 2.5) * E)

drawInf(2., 2.)
drawInf(2.5, 0.)

drawCircle(-2, 0)
drawCircle(1, 0)
drawMark(-1, 0)
drawMark(2, 0)

if draw_dop:
    drawUst(-1, 2)
    drawUst(1.5, 2)
    drawUst(0, 0)
    drawUst(2, -2)
    drawUst(-2, -2)
    drawUst(-2.5, 4.5)

plt.show()

# gamma = 5 Фазовый портрет системы при a=5
gamma = 5.
rhs = f1(gamma)
plotonPlane(rhs, [(-3., 3.), (-5., 5.)])
plt.title("Фазовый портрет системы при a = 5")

# -1 0
drawSep(-1. - 1 * E, 0. + (6 ** 0.5 + 3 ** 0.5 * 2) * E)
drawSep(-1. + 1 * E, 0. - (6 ** 0.5 + 3 ** 0.5 * 2) * E)
drawSep(-1. - 1 * E, 0. + (6 ** 0.5 - 3 ** 0.5 * 2) * E)
drawSep(-1. + 1 * E, 0. - (6 ** 0.5 - 3 ** 0.5 * 2) * E)

# 2 0
drawSep(2. - 1 * E, 0. + (6 ** 0.5 + 3 ** 0.5 * 2) * E)
drawSep(2. + 1 * E, 0. - (6 ** 0.5 + 3 ** 0.5 * 2) * E)
drawSep(2. - 1 * E, 0. + (6 ** 0.5 + 3 ** 0.5 * 2) * E)
drawSep(2. + 1 * E, 0. - (6 ** 0.5 + 3 ** 0.5 * 2) * E)

drawInf(2., 2.)
drawInf(2.5, 0.)

drawCircle(-2, 0)
drawCircle(1, 0)
drawMark(-1, 0)
drawMark(2, 0)

if draw_dop:
    drawUst(-1, 2)
    drawUst(1.5, 2)
    drawUst(0, 0)
    drawUst(2, -2)
    drawUst(-2, -2)
    drawUZ(1 - 1 * E, 0 + 2 * E)
    drawUZ(1 + 1 * E, 0 - 2 * E)
    drawUst(-2.5, 4.5)

plt.show()

# gamma = 4√3 Фазовый портрет системы при a=4√3
gamma = 4. * (3 ** 0.5)
rhs = f1(gamma)
plotonPlane(rhs, [(-3., 3.), (-5., 5.)])
plt.title("Фазовый портрет системы при a = 4√3")

# -1 0
drawSep(-1. - 1 * E, 0. + (3 ** 0.5 * 2 + 2 ** 0.5 * 3) * E)
drawSep(-1. + 1 * E, 0. - (3 ** 0.5 * 2 + 2 ** 0.5 * 3) * E)
drawSep(-1. - 1 * E, 0. + (- 3 ** 0.5 * 2 + 2 ** 0.5 * 3) * E)
drawSep(-1. + 1 * E, 0. - (- 3 ** 0.5 * 2 + 2 ** 0.5 * 3) * E)

# 2 0
drawSep(2. - 1 * E, 0. + (- 2 * 6 ** 0.5 + 3 ** 0.5 * 2) * E)
drawSep(2. + 1 * E, 0. - (- 2 * 6 ** 0.5 + 3 ** 0.5 * 2) * E)
drawSep(2. - 1 * E, 0. + (2 * 6 ** 0.5 + 3 ** 0.5 * 2) * E)
drawSep(2. + 1 * E, 0. - (2 * 6 ** 0.5 + 3 ** 0.5 * 2) * E)

drawInf(2., 2.)
drawInf(2.5, 0.)

drawCircle(-2, 0)
drawCircle(1, 0)
drawMark(-1, 0)
drawMark(2, 0)

if draw_dop:
    drawUst(0, 4)
    drawUst(1, 4)
    drawUst(0, -2)
    drawUst(2, -2)
    drawUst(-2.5, 4)
    drawUZ(-2 - 3 ** 0.5 * E, 0 + 6 * E)
    drawUZ(1 - 1 * E, 0 + (6 ** 0.5 + 3 ** 0.5 * 2) * E)
    drawUZ(1 + 1 * E, 0 - (6 ** 0.5 + 3 ** 0.5 * 2) * E)

plt.show()

# gamma = 10 Фазовый портрет системы при a=10
gamma = 10.
rhs = f1(gamma)
plotonPlane(rhs, [(-3., 3.), (-5., 5.)])
plt.title("Фазовый портрет системы при a = 10")

# -1 0
drawSep(-1. - 1 * E, 0. + (- 31 ** 0.5 + 5) * E)
drawSep(-1. + 1 * E, 0. - (- 31 ** 0.5 + 5) * E)
drawSep(-1. - 1 * E, 0. + (31 ** 0.5 + 5) * E)
drawSep(-1. + 1 * E, 0. - (31 ** 0.5 + 5) * E)

# 2 0
drawSep(2. - 1 * E, 0. + (- 37 ** 0.5 + 5) * E)
drawSep(2. + 1 * E, 0. - (- 37 ** 0.5 + 5) * E)
drawSep(2. - 1 * E, 0. + (37 ** 0.5 + 5) * E)
drawSep(2. + 1 * E, 0. - (37 ** 0.5 + 5) * E)

drawInf(2., 2.)
drawInf(2.5, 0.)

drawCircle(-2, 0)
drawCircle(1, 0)
drawMark(-1, 0)
drawMark(2, 0)

if draw_dop:
    drawUZ(1 - 1 * E, 0 + (19 ** 0.5 + 5) * E)
    drawUZ(1 + 1 * E, 0 - (19 ** 0.5 + 5) * E)
    drawUZ(-2 - 1 * E, 0 + (13 ** 0.5 + 5) * E)
    drawUZ(-2 - 1 / 5, 0 + (-13 ** 0.5 + 5) / 5)
    drawUZ(-2 - 1 * E, 0 + (-13 ** 0.5 + 5) * E)

plt.show()

# gamma = 2, график x(t) при α = 2 в точке (−1.2; 0). Здесь мы можем наблюдать затухающие колебания
gamma = 2.
rhs = f1(gamma)
plotonPlaneWithoutV(rhs, [(-7., 30.), (-5., 10.)])
drawXT(-1.2, 0)
plt.title("График x(t) при α = 2 в точке (-1.2, 0)")

plt.show()

# gamma = 2, график x(t) при α = 2 в точке (−0.8; 0). Здесь мы можем наблюдать затухающие колебания.
gamma = 2.
rhs = f1(gamma)
plotonPlaneWithoutV(rhs, [(-7., 30.), (-5., 10.)])
drawXT(-0.8, 0)
plt.title("График x(t) при α = 2 в точке (-0.8, 0)")

plt.show()

# gamma = 2, график x(t) при α = 2 в точке (2.5; 0). Здесь мы можем наблюдать, как частица приближается к седлу (2; 0), замедляясь около него, а затем движется дальше.
gamma = 2.
rhs = f1(gamma)
plotonPlaneWithoutV(rhs, [(-3., 3.), (0., 20.)])
drawXT(2.5, 0)
plt.title("График x(t) при α = 2 в точке (2.5, 0)")

plt.show()

# gamma = 10, график x(t) при α = 10 в точке (1.5; 0.5)
gamma = 10.
rhs = f1(gamma)
plotonPlaneWithoutV(rhs, [(-3., 3.), (-10., 20.)])
drawXT(1.5, 0.5)
plt.title("График x(t) при α = 10 в точке (1.5, 0.5)")

plt.show()

# gamma = 2√6 график x(t) при α = 2√6 в точке (0; 2). Здесь частица приходит в устойчивый вырожденный узел, оставаясь в нём.
gamma = 2 * 6 ** 0.5
rhs = f1(gamma)
plotonPlaneWithoutV(rhs, [(-3., 3.), (-5., 20.)])
drawXT(0, 2)
plt.title("График x(t) при α = 2√6 в точке (0, 2)")

plt.show()