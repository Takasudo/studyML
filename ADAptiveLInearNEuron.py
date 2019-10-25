import numpy as np

# Page 37

class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output) 
            self.w_[1:] += self.eta * X.T.dot(errors) 
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Page 29 - 30

import pandas as pd
df = pd.read_csv('iris.data',header=None)

X1 = df.iloc[0:100, [0,2]].values
y1_tmp = df.iloc[0:100, 4].values
y1 = np.where(y1_tmp=='Iris-setosa',-1,1)

# Page 40

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X1,y1)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), color='black', marker = 'o',label="$\eta$=0.01 (too large)") 
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum squared error)')  
ax[0].legend()
ada2 = AdalineGD(n_iter=50, eta=0.0001).fit(X1,y1)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, color='black', marker = '^',label="$\eta$=0.0001 (small)")
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('sum squared error')
ax[1].legend()
plt.savefig("page41.pdf")

'''

# Page 32

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0],
                    y=X[y==cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

plt.figure()
plot_decision_regions(X1, y1, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.savefig("page33.pdf")
'''
