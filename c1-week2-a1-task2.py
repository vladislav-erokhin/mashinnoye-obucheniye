import numpy as np
from scipy.linalg import solve
from matplotlib import pylab as plt

def f(x):
    return (np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2))

xx = np.arange(1, 15, 0.1)
yy = f(xx)

# 1 степень
x = np.array([1,15])
y = f(x)

A = np.array([[1,1], [1,15]])
w = solve(A, y)

y1 = w[0] + w[1]*xx

plt.plot(xx, y1, '-', xx, yy, '-')
plt.show()

# 2 степень
x = np.array([1, 8, 15])
y = f(x)

A = np.array([[1,1,1], [1,8,64], [1,15,225]])
w = solve(A, y)

y2 = w[0] + w[1]*xx + w[2]*(xx**2)

plt.plot(xx, y2, '-', xx, yy, '-')
plt.show()

# 3 степень
x = np.array([1, 4, 10, 15])
y = f(x)

A = np.array([[1,1,1,1], [1,4,16,64], [1,10,100,1000], [1,15,225,225*15]])
w = solve(A, y)

y3 = w[0] + w[1]*xx + w[2]*(xx**2) + w[3]*(xx**3)

plt.plot(xx, y3, '-', xx, yy, '-')
plt.show()

print("w 0:4 : ", " ".join(map(str, np.round(w, 2))))