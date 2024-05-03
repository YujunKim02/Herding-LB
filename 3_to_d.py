from itertools import product
from itertools import permutations
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
d = 2
n = 3 ** d
a = [10**i for i in range(d)]
b = [1 for i in range(d)]
b[-1] = n
def num_to_tuple(num):
    return tuple([(num//3**j)%3 for j in range(d)[::-1]])

def tuple_to_num(t):
    num = 0
    d = len(t)
    for i, j in enumerate(t):
        num += 3**(d-i-1) * j
    return num

def generate_vectors(d, a):
    p = [1, -2, 1]
    q = [1, 1, -2]
    v = dict()
    w = dict()
    for num in range(n):
        index = tuple([(num//3**j)%3 for j in range(d)[::-1]])
        v[index] = np.array([a[i]*p[j] for i, j in enumerate(index)])
        w[index] = np.array([a[d-i-1]*p[j] for i, j in enumerate(index)])
    return v, w

def calculate_herd(v_dict, indices):
    summed = np.zeros(d)
    herd = 0
    for index in indices:
        summed += v_dict[index]
        herd = max(herd, np.linalg.norm(summed))
    return herd

def check_id_is_optimal(v_dict, indices, check_times):
    id_herd = calculate_herd(v_dict, indices)
    for _ in range(check_times):
        new_indices = random.sample(indices, k=len(indices))
        perm_herd = calculate_herd(v_dict, new_indices)
        if id_herd > perm_herd:
            return False
    return True

def draw_herd_ratio(v_dict, w_dict, indices, num_samples, threshold = 0):
    v_herds = []
    w_herds = []
    count = 1
    while count < num_samples:
        if count % 100 == 0:
            print(min(v_herds))
        new_indices = random.sample(indices, k=len(indices))
        v_herd = calculate_herd(v_dict, new_indices)
        w_herd = calculate_herd(w_dict, new_indices)
        if threshold == 0:
            v_herds.append(v_herd)
            w_herds.append(w_herd)
            count += 1
        else:
            if v_herd <= threshold:
                v_herds.append(v_herd)
                w_herds.append(w_herd)
                count += 1
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    # sns.kdeplot(x=v_herds, y=w_herds, cmap='viridis', shade=True, shade_lowest=False, cbar=True)
    plt.scatter(v_herds, w_herds, color='blue', label='Herd Values')
    plt.xlabel('v_herd values')
    plt.ylabel('w_herd values')
    plt.legend()
    plt.show()
    return

def count_optimal_perm(indices, v_dict, w_dict, id_herd):
    count = 0
    min_w_herd = calculate_herd(w_dict, indices)
    for perm_indices in tqdm(permutations(indices)):
        if calculate_herd(v_dict, perm_indices) < id_herd:
            min_w_herd = min(min_w_herd, calculate_herd(w_dict, perm_indices))
            count += 1
    return count, min_w_herd

indices = [num_to_tuple(num) for num in range(n)]
check_num = 10000
v, w = generate_vectors(d, a)   # scaling
v2, w2 = generate_vectors(d, b) # no scaling
herd_v = calculate_herd(v, indices)
herd_w = calculate_herd(w, indices)
herd_v2 = calculate_herd(v2, indices)
herd_w2 = calculate_herd(w2, indices)
# draw_herd_ratio(v, w, indices, 100000)
count, min_w_herd = count_optimal_perm(indices, v, w, herd_v + 0.0001)
print(count, min_w_herd)
# draw_herd_ratio(v2, w2, indices, 1000)
print(f"id is optimal over {check_num} number of random permutations:", check_id_is_optimal(v, indices, check_num))
print("Herd value of v:", herd_v)
print("Herd value of w:", herd_w)
print("Ratio herd_w/herd_v:", herd_w/herd_v)
print("d, n:", d, n)