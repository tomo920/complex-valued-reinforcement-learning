import numpy as np
import matplotlib.pyplot as plt

N = 50

def moving_average(l):
    MA = []
    for i in range(len(l)):
        if i+N > len(l):
            break
        Ave = np.average(l[i:i+N])
        MA.append(Ave)
    return MA

result = np.load('result.npy')
# result = result[:250]
MA = moving_average(result)
# MA = result

plt.plot(range(len(MA)), MA)
plt.show()
