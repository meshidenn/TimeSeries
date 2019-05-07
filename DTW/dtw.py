# This Code is quoted from http://ksknw.hatenablog.com/entry/2017/03/26/234048

import numpy as np
import pylab as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from itertools import product


def minVal(v1, v2, v3):
    if extract_value(v1) <= min(extract_value(v2), extract_value(v3)):
        return v1, 0
    elif extract_value(v2) <= extract_value(v3):
        return v2, 1
    else:
        return v3, 2

    
def calc_dtw(A, B):
    S = len(A)
    T = len(B)

    m = [[0 for j in range(T)] for i in range(S)]
    m[0][0] = (norm(A[0], B[0]), (-1, -1))
    for i in range(1, S):
        m[i][0] = (m[i-1][0][0] + norm(A[i], B[0]), (i-1, 0))
    for j in range(1, T):
        m[0][j] = (m[0][j-1][0] + norm(A[0], B[j]), (0, j-1))

    for i in range(1, S):
        for j in range(1, T):
            minimum, index = minVal(m[i-1][j], m[i][j-1], m[i-1][j-1])
            indexes = [(i-1, j), (i, j-1), (i-1, j-1)]
            m[i][j] = (extract_value(minimum) + norm(A[i], B[j]), indexes[index])
    return m


def multi_dtw(data):
    """
    Parameters
    ----------
    data: pandas dataframe
        timeseries index and each timeseries value in column aling

    Returns
    --------
    dtws: dtw values for each two timeseries data
    """

    dtws = []
    
    for i, (key1, key2) in tqdm(enumerate(product(data.columns, data.columns))):
        dtws.append(extract_value(calc_dtw(data[key1].get_values(),
                                           data[key2].get_values())[-1][-1]))

    dtws = np.array(dtws).reshape(len(data.columns), -1)
    return dtws
     

def extract_value(value_index_pair):
    return value_index_pair[0]


def extract_index(value_index_pair):
    return value_index_pair[1]


def norm(a, b):
    return (a - b)**2


def backward(m):
    path = []
    path.append([len(m)-1, len(m[0])-1])
    while True:
        path.append(extract_index(m[path[-1][0]][path[-1][1]][1]))
        if path[-1] == (0, 0):
            break
    path = np.array(path)
    return path


def plot_path(m, path, A, B, filename=""):
    """
    this plot can use only when you plot 2 time series data
    """
    
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 5],
                           height_ratios=[5, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax4 = plt.subplot(gs[3])
    
    list_delta = [[extract_value(t) for t in row] for row in m]
    list_delta = np.array(list_delta)
    ax2.pcolor(list_delta, cmap=plt.cm.Blues)
    ax2.plot(path[:, 1], path[:, 0], c="C3")
    
    ax1.plot(A, range(len(A)))
    ax1.invert_xaxis()
    ax4.plot(B, c="C1")
    plt.show()
    
    for line in path:
        plt.plot(line, [A[line[0]], B[line[1]]], linewidth=0.2, c="gray")
    plt.plot(A)
    plt.plot(B)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    
