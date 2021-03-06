import pandas as pd
from matplotlib import pyplot as plt

df1 = pd.read_csv("cnnet_fusion_cifar_separated_mp.csv")
df2 = pd.read_csv("cnnet_fusion_separated_mp.csv")
df3 = pd.read_csv("cnnet_fusion_cifar_jointed_mp.csv")
df4 = pd.read_csv("cnnet_fusion_jointed_mp.csv")
dataname = "Prediction in AUC and degree"

fig = plt.figure(figsize=(10, 6))
fig.suptitle(dataname)
# line1, = plt.plot(df1.AUC, 'g-', label='kernels pruned separated, CIFAR')
# plt.plot(df1.as_matrix()[:5, 2], df1.AUC[df1.as_matrix()[:5, 2]], 'g-')
line1, = plt.plot(df1.as_matrix()[:5, 2], df1.as_matrix()[:5, 1], 'g-o', label='kernels pruned separated, CIFAR')


# line2, = plt.plot(df2.AUC, 'r-', label='kernels pruned separated, MNIST')
# plt.plot(df2.as_matrix()[:5, 2], df2.AUC[df2.as_matrix()[:5, 2]], 'r-')
line2, = plt.plot(df2.as_matrix()[:5, 2], df2.as_matrix()[:5, 1], 'r-o', label='kernels pruned separated, MNIST')


# line3, = plt.plot(df3.AUC, 'b-', label='kernels pruned jointed, CIFAR')
# plt.plot(df3.as_matrix()[:5, 2], df3.AUC[df3.as_matrix()[:5, 2]], 'b-')
line3, = plt.plot(df3.as_matrix()[:5, 2], df3.as_matrix()[:5, 1], 'b-o', label='kernels pruned jointed, CIFAR')


# line4, = plt.plot(df4.AUC, 'c-', label='kernels pruned jointed, MNIST')
# plt.plot(df4.as_matrix()[:5, 2], df4.AUC[df4.as_matrix()[:5, 2]], 'c-')
line4, = plt.plot(df4.as_matrix()[:5, 2], df4.as_matrix()[:5, 1], 'c-o', label='kernels pruned jointed, MNIST')

plt.xlabel('Iteration')
plt.ylabel('Degree')
# plt.legend(loc=4)
leg1 = plt.legend(handles=[line1, line2, line3, line4], loc=0)
fig.savefig(dataname + '_cnnet.png')
# plt.show()

# dataname = "Prediction time in degree"
# fig = plt.figure(figsize=(20, 5))
# fig.suptitle(dataname)
# # plt.plot(df2.TIME, 'b-', label='sep at %.2f seconds, MNIST' % df2.reference[0])
# # plt.plot(df3.TIME, 'm-', label='join at %.2f seconds, CIFAR' % df3.reference[0])
# # plt.plot(df4.TIME, 'r-', label='join at %.2f seconds, MNIST' % df4.reference[0])
# line1, = plt.plot(df1.TIME, 'g-', label='kernels pruned separated, CIFAR, reference time: %.3f'%df1.reference[0])
# plt.plot(df1.as_matrix()[:5, 2], df1.TIME[df1.as_matrix()[:5, 2]], 'go')
# plt.text(df1.as_matrix()[0, 2], df1.TIME[df1.as_matrix()[0, 2]], r'$%.2f$'%df1.as_matrix()[0, 1])
# plt.text(df1.as_matrix()[1, 2], df1.TIME[df1.as_matrix()[1, 2]], r'$%.2f$'%df1.as_matrix()[1, 1])
# plt.text(df1.as_matrix()[2, 2], df1.TIME[df1.as_matrix()[2, 2]], r'$%.2f$'%df1.as_matrix()[2, 1])
# plt.text(df1.as_matrix()[3, 2], df1.TIME[df1.as_matrix()[3, 2]], r'$%.2f$'%df1.as_matrix()[3, 1])
# plt.text(df1.as_matrix()[4, 2], df1.TIME[df1.as_matrix()[4, 2]], r'$%.2f$'%df1.as_matrix()[4, 1])
#
# line2, = plt.plot(df2.TIME, 'r-', label='kernels pruned separated, MNIST, reference time: %.3f'%df2.reference[0])
# plt.plot(df2.as_matrix()[:5, 2], df2.TIME[df2.as_matrix()[:5, 2]], 'ro')
# plt.text(df2.as_matrix()[0, 2], df2.TIME[df2.as_matrix()[0, 2]], r'$%.2f$'%df2.as_matrix()[0, 1])
# plt.text(df2.as_matrix()[1, 2], df2.TIME[df2.as_matrix()[1, 2]], r'$%.2f$'%df2.as_matrix()[1, 1])
# plt.text(df2.as_matrix()[2, 2], df2.TIME[df2.as_matrix()[2, 2]], r'$%.2f$'%df2.as_matrix()[2, 1])
# plt.text(df2.as_matrix()[3, 2], df2.TIME[df2.as_matrix()[3, 2]], r'$%.2f$'%df2.as_matrix()[3, 1])
# plt.text(df2.as_matrix()[4, 2], df2.TIME[df2.as_matrix()[4, 2]], r'$%.2f$'%df2.as_matrix()[4, 1])
#
# line3, = plt.plot(df3.TIME, 'b-', label='kernels pruned jointed, CIFAR, reference time: %.3f'%df3.reference[0])
# plt.plot(df3.as_matrix()[:5, 2], df3.TIME[df3.as_matrix()[:5, 2]], 'bo')
# plt.text(df3.as_matrix()[0, 2], df3.TIME[df3.as_matrix()[0, 2]], r'$%.2f$'%df3.as_matrix()[0, 1])
# plt.text(df3.as_matrix()[1, 2], df3.TIME[df3.as_matrix()[1, 2]], r'$%.2f$'%df3.as_matrix()[1, 1])
# plt.text(df3.as_matrix()[2, 2], df3.TIME[df3.as_matrix()[2, 2]], r'$%.2f$'%df3.as_matrix()[2, 1])
# plt.text(df3.as_matrix()[3, 2], df3.TIME[df3.as_matrix()[3, 2]], r'$%.2f$'%df3.as_matrix()[3, 1])
# plt.text(df3.as_matrix()[4, 2], df3.TIME[df3.as_matrix()[4, 2]], r'$%.2f$'%df3.as_matrix()[4, 1])
#
# line4, = plt.plot(df4.TIME, 'c-', label='kernels pruned jointed, MNIST, reference time: %.3f'%df4.reference[0])
# plt.plot(df4.as_matrix()[:5, 2], df4.TIME[df4.as_matrix()[:5, 2]], 'co')
# plt.text(df4.as_matrix()[0, 2], df4.TIME[df4.as_matrix()[0, 2]], r'$%.2f$'%df4.as_matrix()[0, 1])
# plt.text(df4.as_matrix()[1, 2], df4.TIME[df4.as_matrix()[1, 2]], r'$%.2f$'%df4.as_matrix()[1, 1])
# plt.text(df4.as_matrix()[2, 2], df4.TIME[df4.as_matrix()[2, 2]], r'$%.2f$'%df4.as_matrix()[2, 1])
# plt.text(df4.as_matrix()[3, 2], df4.TIME[df4.as_matrix()[3, 2]], r'$%.2f$'%df4.as_matrix()[3, 1])
# plt.text(df4.as_matrix()[4, 2], df4.TIME[df4.as_matrix()[4, 2]], r'$%.2f$'%df4.as_matrix()[4, 1])
# plt.xlabel('iteration')
# plt.ylabel('prediction time')
# plt.axis([0, 120, 0.78, 1.04])
# leg1 = plt.legend(handles=[line1, line2, line3, line4], loc=3)
# fig.savefig(dataname + '_cnnet.png')
# plt.show()