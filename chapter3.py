from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1],c='', alpha=1.0, linewidths=1, marker='o',s=55)



iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

#(0,1,2) the various type of Iris
#print(np.unique(y))

#Random divide the dataset in test and train set
#In this case test_size 30%
X_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

sc= StandardScaler()
#Calculate sample median and standard deviation
#For each dimension in this case 2
sc.fit(X_train)

#standardization thanks to the previous fit
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(x_test)

#inizializzazione del classificatore
ppn = Perceptron(n_iter=40, alpha=0.1, random_state=0)
#Addestramento
ppn.fit(X_train_std, y_train)
#After the training phase we can predict
y_pred = ppn.predict(X_test_std)
print('Misclassified sample: %d' %(y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#Perceptron

X_combined_std =np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()

#Logistic Regression

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=lr,test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal with')
plt.legend(loc='upper left')
plt.show()
print(lr.predict_proba(X_test_std[0]))