# The function that we optimize is $F(x)=\frac{1}{n}\sum_{i=1}^n f_i(x)$, where $f_i(x) = x^2 - x$ if $i\leq n/2$, and $f_i(x) = x$ otherwise.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
# Hyperparameters
sF=1
mu = 0.01
L = 1
kappa = L/mu
n = 10

# Logarithmic scale range for step size
K_beg=10
K_end=301
res = np.zeros((K_end-1, n))
K_range = range(K_beg,K_end,1)

start = 1
stop = 0.01
num_points = 50
# step_sizes = np.logspace(np.log10(start), np.log10(stop), num=num_points)
step_sizes = np.linspace(start, stop, num_points)

def parallelOptimization_K():
    
    errors = []
    n_values = []
    for n in range(4, 20, 2):
        K = int(100)
        eta = 1 / (mu * n * K)
        error = IGD(eta, n, K)
        errors.append(error)
        n_values.append(n)
    return errors, n_values

def parallelOptimization(K):
  eta = np.log(n*K) / (mu*n*K)
  error = IGD(eta, n, K)
  return [error, K]

def parallelOptimization_all():
  res = []
  for i in range(len(K_range)):
    K = K_range[i]
    
    for j in range(num_points):
      eta = step_sizes[j]
      error = IGD(eta, n, K)
      error = np.log(error)
      res.append((error, K, eta))
  return res

def parallelOptimization_good_eta():
  res = []
  for i in range(len(K_range)):
    K = K_range[i]
    for j in range(num_points):
      eta = 1 / (mu*n*K)
      error = IGD(eta, n, K)
      error = np.log(error)
      res.append((error, K, eta))
  return res

def parallelOptimization_good_eta2():
  res = []
  for i in tqdm(range(len(K_range))):
    K = K_range[i]
    for j in range(num_points):
      eta = np.log(n*K) / (mu*n*K)
      error = IGD(eta, n, K)
      error = np.log(error)
      res.append((error, K, eta))
  return res

def IGD(eta, n, K):
  # x=0+np.random.normal(0,1)
  x = 1
  for i in range(1, K+1):  
    for j in range(0, int(n/2)):
      # x = x - eta # The gradient step.
      x = x - eta * (x + 1)

    for j in range(int(n/2), n):
      # x = x - eta*(2*x - 1) # The gradient step.
      x = x - eta * (3*x - 1)
  return x**2 # Error is x**2, since the minimizer is x=0

# Plot the result
errors, n_values = parallelOptimization_K()
plt.plot(n_values, errors)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('Error')
plt.title('Error vs n, K = 100, eta = 1/mu * n * K')

# Plot the line with slope -1
x_values = np.array(n_values)
y_values_line = 1 / (x_values ** 1) # Assuming y-intercept as 10
plt.plot(x_values, y_values_line, label='Slope -1')
plt.legend()
plt.grid(True, linewidth=0.5)
plt.show()
# results=[]
# for k in K_range:
#   results.append(parallelOptimization(k))

# f = open('plotdatanew/IGD_3d', 'w') # Replace with desired output file name
# f.write("\n".join([",".join([str(r) for r in res]) for res in results]))

# Plot 3d Data
res_all = parallelOptimization_all()
res_good = parallelOptimization_good_eta()
res_good2 = parallelOptimization_good_eta2()
error_values, k_values, eta_values = zip(*res_all)
good_error_values, good_k_values, good_eta_values = zip(*res_good)
good_error_values2, good_k_values2, good_eta_values2 = zip(*res_good2)


f = open('plotdatanew/IGD_component_sc', 'w')
results = list(zip(good_error_values, good_k_values))
f.write("\n".join([",".join([str(r) for r in res]) for res in results]))

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(eta_values, k_values, error_values, c='r', marker='.')
# ax.scatter(good_eta_values, good_k_values, good_error_values, c='b', marker='.', label = '1/mu n k')
# # ax.scatter(good_eta_values2, good_k_values2, good_error_values2, c='g', marker='.', label = 'log(nk)/mu n k')

# ax.set_xlabel('eta')
# ax.set_ylabel('k')
# ax.set_zlabel('Error(log)')
# plt.legend
# plt.show()