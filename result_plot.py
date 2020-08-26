import numpy as np
import matplotlib.pyplot as plt

N = 5000

def moving_average(l):
    MA = []
    for i in range(len(l)):
        if i < N-1:
            MA.append(l[i])
        else:
            Ave = np.average(l[i+1-N:i+1])
            MA.append(Ave)
    return MA

result = np.load('result.npy')
result = result[:250]
MA = moving_average(result)

plt.plot(range(len(MA)), MA)
plt.show()
