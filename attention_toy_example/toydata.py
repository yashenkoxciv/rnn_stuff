import numpy as np
import matplotlib.pyplot as plt

X_DIM = 80
POS = 1

def next_batch(batch_size, dim, pos):
    y = np.random.randint(0, 2, batch_size)
    #x = np.random.randn(batch_size, dim)*0.5 + 1.
    x = np.random.uniform(size=[batch_size, dim])
    x[:, pos] = y
    return x, y.reshape([-1, 1])

if __name__ == '__main__':
    X_DIM = 10
    x, y = next_batch(30, X_DIM, POS)
    
    pidx = np.argwhere(y.flatten() == 1)
    for i in pidx:
        plt.scatter(np.arange(0, X_DIM), x[i].flatten())
    plt.plot(POS, 1, marker='X', markersize=10)
    plt.show()
    
    nidx = np.argwhere(y.flatten() == 0)
    for i in nidx:
        plt.scatter(np.arange(0, X_DIM), x[i].flatten())
    plt.plot(POS, 0, marker='o', markersize=10)
    plt.show()

