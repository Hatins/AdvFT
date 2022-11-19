import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

#Note that the data are come from the experiments
y1 = [13.86,34.67,54.01,11.35,11.31,11.30,11.35,11.35,11.35,11.32,11.30]
x1 = [1,3,5,7,10,20,30,40,50,75,100]

y2 = [77.42,65.48,61.88,52.29,38.96,54.06,14.69,13.17,20.86,22.53,26.21]
x2 = [1,3,5,10,20,30,40,50,75,83,100]

y3 = [52.75,36.68,25.51,36.30,13.01,16.57,15.77,24.63,16.56,18.16,10]
x3 = [1,3,5,10,20,30,40,50,72,75,99]

plt.style.use('science')
fig, ax = plt.subplots(figsize=(4, 3),dpi=600)
p1 = plt.scatter(x1, y1, marker='o',s=8,color=sns.xkcd_rgb["bright orange"], label='MNIST' + '-' + 'LeNet (data point)')
p2 = plt.scatter(x2, y2, marker='o',s=8,color=sns.xkcd_rgb["dark blue"], label='FMNIST' + '-' + 'ResNet18 (data point)')
p3 = plt.scatter(x3, y3, marker='o',s=8,color=sns.xkcd_rgb["teal"], label='CIFAR10' + '-' + 'ResNet18 (data point)')
y1 = gaussian_filter1d(y1, sigma=1)
y2 = gaussian_filter1d(y2, sigma=1)
y3 = gaussian_filter1d(y3, sigma=1)
line1, = plt.plot(x1, y1, linestyle='--', color=sns.xkcd_rgb["bright orange"], label='MNIST' + '-' + 'LeNet (fitted line)', lw=1)
line2, = plt.plot(x2, y2, linestyle='--', color=sns.xkcd_rgb["dark blue"], label='FMNIST' + '-' + 'ResNet18 (fitted line)', lw=1)
line3, = plt.plot(x3, y3, linestyle='--', color=sns.xkcd_rgb["teal"], label='CIFAR10' + '-' + 'ResNet18 (fitted line)', lw=1)
plt.legend(handles=[line1,p1, line2,p2, line3,p3], loc='upper right',fontsize=7)
ax.set_xlabel(r'$epoch$',fontsize=8)
ax.set_ylabel(r'Accuracy of $\mathcal{M}_S$ (\%)',fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()
