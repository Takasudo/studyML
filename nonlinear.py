# Common : dataset

from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # calculate mean and deviation for X_train
X_train_std = sc.transform(X_train) 
X_test_std = sc.transform(X_test)

# Common : plot_decision_regions

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx = None, resolution=0.02):
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

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1],
                    c='',
                    edgecolor = 'black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')

# Common : combined std data

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Page 81 

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:,0] >0, X_xor[:, 1]>0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1], c='blue', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1], c='red', marker='s', label='-1')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.legend()
plt.tight_layout()
plt.savefig("page82.pdf")

from sklearn.svm import SVC

svm = SVC(kernel = 'rbf', gamma = 0.1, C=1.0, random_state=1)
svm.fit(X_xor, y_xor)
plt.figure()
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend()
plt.tight_layout()
plt.savefig("page85.pdf")

svm2 = SVC(kernel = 'rbf', gamma = 10., C=1.0, random_state=1)
svm2.fit(X_xor, y_xor)
plt.figure()
plot_decision_regions(X_xor, y_xor, classifier=svm2)
plt.legend()
plt.tight_layout()
plt.savefig("page85_large_gamma.pdf")
