import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8',  '#4daf4a','#ff7f00',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']) 

matplotlib.rc('lines', linewidth=2)


f = open('plotdatanew/IGD_component_sc', 'r') # Replace with desired input file name
output = f.read()
output = output.split("\n")
# results = output[1:]
results = output[:]
e1 = [float(res.split(",")[0]) for res in results]
x_list = [float(res.split(",")[1]) for res in results]

# l1 = [np.exp(x)/np.exp(e1[0]) for x in e1]
l1 = [np.exp(x) for x in e1]
r1=[(x_list[0]**1)/(x**1) for x in x_list]
r2=[(x_list[0]**2)/(x**2) for x in x_list]

plt.plot(x_list, l1, linestyle='solid', label=r'IGD')
plt.plot(x_list, r1, linestyle='dashed', label=r'$1/K^1$')
plt.plot(x_list, r2, linestyle='dashed', label=r'$1/K^2$')

plt.yscale('log')
plt.xscale('log')
plt.title('eta = 1/mu*n*k')
plt.legend(loc='lower left',fontsize=18,ncol=2)
plt.xlabel(r'Number of epochs $K$',fontsize=18)
plt.ylabel(r'error (Not Normalized)',fontsize=18)

ax1 = plt.gca()
ax1.set_xticks([2, 4, 10, 30, 40, 60, 90, 150])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.savefig('IGD.pdf')  

