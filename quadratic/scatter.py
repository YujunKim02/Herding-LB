import numpy as np
import matplotlib.pyplot as plt

# Define values for k
k_values = np.arange(10, 101)

# Calculate the expression for each value of k
n = 10
mu = 0.01
expr_values1 = np.log(n * k_values) / (mu * n * k_values)
expr_values2 = 1 / (mu * n * k_values)

# Plot the values
plt.plot(k_values, expr_values1, label='log(nk) / (mu * nk)')
plt.plot(k_values, expr_values2, label='1 / (mu * nk)')
plt.xlabel('k')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
