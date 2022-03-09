# #skrypt 1
#
import numpy as np
import matplotlib.pyplot as plt
#
# x_1 = np.array([0,1])
# x_2 = np.array([1,0])
# x_3 = np.array([0,0])
# x_4 = np.array([1,1])
#
# X = np.array([x_1,x_2,x_3,x_4])
# y = np.array([0,0,1,1])
#
# print('x',x_1,x_1.shape)
# print('x',X,X.shape)
# print('y',y,y.shape)
#
# fig, ax = plt.subplots(1,1,figsize=(5,5/1.618))
# # foo = np.linspace(0,1,4)
#
# # print(foo,y)
#
# # ax[1].plot(y)
# # print('X[:,0]',X[:,0])
# # print('X[:,1]',X[:,1])
#
#
#
# ax.scatter(X[:,0],
#            X[:,1], c=y, cmap='bwr', s=100)

# plt.savefig('foo.png')

#skrypt 2
from sklearn.datasets import make_classification

cc ={
    'n_features': 2,
    'n_informative': 2,
    'n_redundant': 0, #nadmiarowe
    'n_repeated': 0, #liczba atrybutów powtórzonych
    'n_samples': 50, #liczba wzorców
    'random_state': 1024 #ziarno losowości
}


X, y = make_classification(**cc)


# print(y)
mask_negative = y==0
mask_positive = y==1

# print('X', X.shape)
# print('Xn', X[mask_negative].shape)
# print('Xp', X[mask_positive].shape)

exit()
# print('x',X[0],X[0].shape)
# print('x',X,X.shape)
# print('y',y,y.shape)
#
# print('X[:,0]',X[:,0])
# print('X[:,1]',X[:,1]

# y =ax + b

def activation_function(output):
    return output > 0

def decision_function(X):
    weight = np.array([-1,2]) # wsp. kier
    bias = np.array([0,1]) # wyraz wolny

    return activation_function(np.sum(X*weight - bias, axis=1))

y_pred = decision_function(X)

# print(y[10])
# print(y_pred[10])

fig, ax = plt.subplots(2,2,figsize=(8,8))

ax[0,0].scatter(X[:,0],X[:,1],c=y, cmap='bwr')
ax[0,0].set_title('Dataset')
ax[0,0].set_xlabel('feature 0')
ax[0,0].set_ylabel('feature 1')

ax[0,1].scatter(X[:,0],X[:,1],c=y, cmap='bwr')
ax[0,1].set_title('Dataset')
ax[0,1].set_xlabel('feature 0')
ax[0,1].set_ylabel('feature 1')

# ax[0,0].scatter(X[:,0],X[:,1],c=y, cmap='bwr')
# ax[0,0].set_title('Dataset')
# ax[0,0].set_xlabel('feature 0')
# ax[0,0].set_ylabel('feature 1')

ax[1,0].boxplot([
    X[mask_negative,0],
    X[mask_positive,0]
],labels=['negative','positive'])

ax[1,1].boxplot([
    X[mask_negative,1],
    X[mask_positive,1]
],labels=['negative','positive'])

# ax[0,1].scatter(X[:,0],X[:,0],c=y, cmap='bwr')
# ax[0,1].set_title('Dataset')
# ax[0,1].set_xlabel('feature 0')
# ax[0,1].set_ylabel('feature 0')
#
# ax[1,1].scatter(X[:,1],X[:,1],c=y, cmap='bwr')
# ax[1,1].set_title('Dataset')
# ax[1,1].set_xlabel('feature 1')
# ax[1,1].set_ylabel('feature 1')


plt.tight_layout()
plt.savefig('foo.png')