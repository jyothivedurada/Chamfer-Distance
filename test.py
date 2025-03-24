# Testing script for verification purposes
# To use, set the values of M, N and DIMS here
M = 10
N = 20
DIMS = 64
# Then, run the main program with the debug print statements in the main function uncommented
# And give the output of the program as input to this program

import numpy as np

m = np.ndarray((M, DIMS), dtype=np.float32)
n = np.ndarray((N, DIMS), dtype=np.float32)

for v in range(m.shape[0]):
    for i in range(m.shape[1]):
        m[v, i] = float(input().split()[3])

for v in range(n.shape[0]):
    for i in range(n.shape[1]):
        n[v, i] = float(input().split()[3])

# Calculate the chamfer distances between m and n
# and print the result

def chamfer_distance(m, n):
    dist = 0
    for i in range(m.shape[0]): dist += np.sum((n - m[i])**2, axis=1).min()
    for i in range(n.shape[0]): dist += np.sum((m - n[i])**2, axis=1).min()

    return dist

result = chamfer_distance(m, n)
print("Chamfer Distance:", result)