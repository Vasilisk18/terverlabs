from os import name
from math import pi, sin
from pickle import TRUE
from tkinter.ttk import Scale
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# L = 1; L1 = 2; L2 = 1
# C1 = 0.7; C2 = 0.5


# L = 1; L1 = 2; L2 = 1
# C1 = 2; C2 = 2

L = 0.5;
L1 = 5;
L2 = 0.8
C1 = 1;
C2 = 1


def plotonPlaneWithoutV(limits):
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlabel('w1')
    ax.set_ylabel('W')

    # fig.set_size_inches(17, 10)   
    xlims, ylims = limits
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])


plotonPlaneWithoutV([(0., 5.), (0., 5.)])

x = []
y1 = []
y2 = []
step = 0.01

w1 = (L2 + L) / (C1 * (L1 + L) * (L2 + L) - L * L)
w2 = (L1 + L) / (C2 * (L1 * L + L1 * L2 + L2 * L))
a = L * L / (C1 * C2 * (L1 * L + L1 * L2 + L2 * L))
print(w1, w2, a)
# w1 = 0; w2 = 3; a = 0.01

for i in range(0, 1000):
    w1 = (L2 + L) / (C1 * (L1 + L) * (L2 + L) - L * L)
    w2 = (L1 + L) / (C2 * (L1 * L + L1 * L2 + L2 * L))

    a = L * L / (C1 * C2 * (L1 * L + L1 * L2 + L2 * L))

    W1 = (0.5 * (w1 + w2 + ((w2 - w1) ** 2 + 4 * a) ** 0.5)) ** 0.5
    W2 = (0.5 * (w1 + w2 - ((w2 - w1) ** 2 + 4 * a) ** 0.5)) ** 0.5
    x.append(w1)
    y1.append(W1)
    y2.append(W2)

    # w1+=step
    L1 -= step
    if (w1 >= 10):
        break

plt.plot(x, y1, 'r-')
plt.plot(x, y2, 'b-')
plt.show()