import numpy as np
import math
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from matplotlib import pylab as plt


# Задача 1

def f(x):
    return np.sin(x / 5.) * np.exp(x / 10.) + 5. * np.exp(-x / 2.)


def f_scalar(x):
    return math.sin(x / 5.) * math.exp(x / 10.) + 5. * math.exp(-x / 2.)


x = np.arange(1., 30., 0.1)

plt.plot(x, f(x))

# Попробуйте найти минимум, используя стандартные параметры в функции
# Попробуйте менять начальное приближение и изучить, меняется ли результат
print(minimize(f_scalar, 1.).fun)
print(minimize(f_scalar, 5.).fun)
print(minimize(f_scalar, 13.).fun)

bfgsMin2 = minimize(f_scalar, 2., method='BFGS')
print("BFGS из точки 2: ", round(bfgsMin2.fun, 2))

bfgsMin30 = minimize(f_scalar, 30., method='BFGS')
print("BFGS из точки 30: ", round(bfgsMin30.fun, 2))

with open('c1-week3-a1\\1.txt', 'w') as f1:
    f1.write(str(round(bfgsMin2.fun, 2)) + " " + str(round(bfgsMin30.fun, 2)))

# Задача 2
deMin = differential_evolution(f_scalar, [(1., 30.)])

print("Итераций для BFGS: ", bfgsMin30.nit)
print("Итераций для differential_evolution: ", deMin.nit)

with open('c1-week3-a1\\2.txt', 'w') as f2:
    f2.write(str(round(deMin.fun, 2)))

# Задача 3
def h_scalar(x):
    return int(f_scalar(x))

def h(x):
    return np.rint(f(x))

plt.plot(x, h(x))

h_bfgsMin30 = minimize(h_scalar, 30., method='BFGS')
print("BFGS из точки 30 для h(x): ", round(h_bfgsMin30.fun, 2))

h_deMin = differential_evolution(h_scalar, [(1., 30.)])
print("differential_evolution из точки 30 для h(x): ", round(h_deMin.fun, 2))

with open('c1-week3-a1\\3.txt', 'w') as f1:
    f1.write(str(round(h_bfgsMin30.fun, 2)) + " " + str(round(h_deMin.fun, 2)))
