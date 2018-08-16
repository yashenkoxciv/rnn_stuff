import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def sin_data(num, ascending=False):
    #x = np.linspace(-np.pi, np.pi, num)
    x = np.linspace(0, 10*np.pi, num)
    x = x + np.abs(x.min())
    if ascending:
        y = 5*np.sin(x) + x
    else:
        y = np.sin(x)
    x = x / (x.max() + 1e-5)
    y = y + np.abs(y.min())
    y = y / (y.max() + 1e-5)
    return x, y

def show_examples(nums):
    x, y = sin_data(nums)    
    plt.plot(*sin_data(200))
    plt.scatter(x, y)
    plt.show()

def next_batch(x, unrollings, batch_size):
    s_idx = np.random.randint(0, x.shape[0]-unrollings-1, batch_size)
    batch_x = []
    batch_y = []
    for s in s_idx:
        #batch_x.append(x[s:s+unrollings].reshape([-1, 1]))
        #batch_y.append(x[s+unrollings].reshape([1, -1]))
        batch_x.append(x[s:s+unrollings])
        batch_y.append(x[s+unrollings])
    return np.array(batch_x), np.array(batch_y)

def next_unrolling(x, unrollings):
    s = np.random.randint(0, x.shape[0]-unrollings-1)
    return x[s:s+unrollings], x[s+unrollings].reshape([1])

if __name__ == '__main__':
    show_examples(10)
    xaxis, x = sin_data(10)
    batch_x, batch_y = next_batch(x, 3, 2)
    print('Data', x, sep='\n')
    print('Batch', batch_x, batch_y, sep='\n')
