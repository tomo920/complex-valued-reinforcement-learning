import numpy as np
import matplotlib.pyplot as plt

N = 2000

def moving_average(l):
    MA = []
    for i in range(len(l)):
        if i < N-1:
            MA.append(l[i])
        else:
            Ave = np.average(l[i+1-N:i+1])
            MA.append(Ave)
    return MA

result_q = np.load('q_learning.npy')
result_qdot_1 = np.load('result_qdot_Ne_1.npy')
result_qdot_2 = np.load('result_qdot_Ne_2.npy')
result_q = result_q[:200000]
result_qdot_1 = result_qdot_1[:200000]
result_qdot_2 = result_qdot_2[:200000]
MA_q = moving_average(result_q)
MA_qdot_1 = moving_average(result_qdot_1)
MA_qdot_2 = moving_average(result_qdot_2)

x = range(len(MA_qdot_1))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(range(len(MA_q)), MA_q, label = "Q-learning")
ax.plot(range(len(MA_qdot_1)), MA_qdot_1, label = "$\dot{\mathrm{Q}}$-learning(Ne=1)")
ax.plot(range(len(MA_qdot_2)), MA_qdot_2, label = "$\dot{\mathrm{Q}}$-learning(Ne=2)")
ax.grid(axis='both')

# 凡例の表示
ax.legend()
plt.show()
