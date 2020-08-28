import numpy as np
import matplotlib.pyplot as plt

N = 10

def moving_average(l):
    MA = []
    for i in range(len(l)):
        if i < N-1:
            MA.append(l[i])
        else:
            Ave = np.average(l[i+1-N:i+1])
            MA.append(Ave)
    return MA

result_q = np.load('result_q.npy')
result_qdot_6 = np.load('result_qdot_Ne_6.npy')
result_qdot_1 = np.load('result_qdot_Ne_1.npy')
result_rw = np.load('randomwalk.npy')
result_q = result_q[:2000]
result_qdot_6 = result_qdot_6[:2000]
result_qdot_1 = result_qdot_1[:2000]
result_rw = result_rw[:2000]
MA_q = moving_average(result_q)
MA_qdot_6 = moving_average(result_qdot_6)
MA_qdot_1 = moving_average(result_qdot_1)
MA_rw = moving_average(result_rw)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(range(len(MA_qdot_6)), MA_qdot_6, label = "$\dot{\mathrm{Q}}$-learning(Ne=6)")
ax.plot(range(len(MA_qdot_1)), MA_qdot_1, label = "$\dot{\mathrm{Q}}$-learning(Ne=1)")
ax.plot(range(len(MA_rw)), MA_rw, label = "Random walk")
ax.plot(range(len(MA_q)), MA_q, label = "Q-learning")
ax.grid(axis='both')

# 凡例の表示
ax.legend()
plt.show()
