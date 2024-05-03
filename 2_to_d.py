import numpy as np
from itertools import permutations
from tqdm import tqdm

def generate_x(d):
    if d == 1:
        return [[1], [-1]]
    xs = generate_x(d-1)
    # print(xs)
    ys = []
    for i, x in enumerate(xs):
        new_x = x.copy()
        new_x.append((-1) ** i)
        # print("new_x", new_x)
        ys.append(new_x)
    for i, x in enumerate(xs[::-1]):
        mx = [-1*e for e in x]
        mx.append((-1) ** (i+1))
        # print("mx", mx)
        ys.append(mx)
    return ys

def generate_v(xs):
    d = len(xs[0])
    zeros = np.zeros(d)
    xs.insert(0, zeros)
    xs.insert(-1, zeros)
    xs = np.array(xs)
    vs = []
    # print(xs.shape)
    for i in range(len(xs) - 1):
        vs.append(xs[i+1] - xs[i])
    return vs

def count_elem(input_list):
    tuple_list = [tuple(_) for _ in input_list]
    return len(tuple_list)

def max_cum_sum(input_list):
    d = len(input_list[0])
    max_cum_sum = 0
    cum_sum = np.zeros(d)
    for elem in input_list:
        cum_sum += elem
        max_cum_sum = max(max_cum_sum, np.linalg.norm(cum_sum))
    return max_cum_sum

def max_cum_sum_start(input_list, start): # start(for computational efficiency)
    d = len(input_list[0])
    max_cum_sum = 0
    cum_sum = start
    for elem in input_list:
        cum_sum += elem
        max_cum_sum = max(max_cum_sum, np.linalg.norm(cum_sum))
    return max_cum_sum

def calculate_herd(vs):
    herd = max_cum_sum(vs)
    count = 0
    for perm in tqdm(permutations(vs)):
        herd_perm = max_cum_sum(perm)
        if herd_perm < herd:
            count = 1
            herd = herd_perm
        elif herd_perm == herd:
            count += 1
    return count, herd

def count_optimal_perm_fast(vs, cum_sum): # Assume optimal herd = sqrt(d)(1 in infinity norm)
    """
    Input
    cum_sum: cumulative sum so far
    vs: remaining vectors to test

    Return
    Consider all permutation 'perm' over all permutation on vs.
    Return number of 'perm' so that max_cum_sum_start(perm, start) <= 1
    
    If cum_sum has infinity norm greater than 1, return 0(Cut of current node of search tree)
    Else if vs is empty and cumulative sum so far has infinity norm <= 1, return 1(Reached successful leaf)
    Else, add count over all child node
    """
    cum_sum_inf_norm = np.linalg.norm(cum_sum, np.inf)
    if cum_sum_inf_norm > 1:
        return 0
    if vs == [] and cum_sum_inf_norm <= 1:
        return 1
    counts = []
    for i, v in enumerate(vs):
        new_vs = vs.copy()
        v = new_vs.pop(i)
        counts.append(count_optimal_perm_fast(new_vs, cum_sum + v))
    return sum(counts)

d = 9
xs = generate_x(d)
vs = generate_v(xs)

# count, herd = calculate_herd_fast(vs)
# print(f"Number of permutation attained optimal herd: {count}, where herd is {herd}")
count = count_optimal_perm_fast(vs, np.zeros(d))
print(count)