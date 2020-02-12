import numpy as np
import math
from sklearn.svm import SVC
import csv
import matplotlib.pyplot as plt

def init_data(t):
    data = []
    with open('svm_dataset\\' + t + '.csv') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader) # take the header out
        for row in reader: # each row is a list
            data.append(row)
    data  = np.array(data, dtype = np.float)
    np.random.shuffle(data)
    X = data[:,:-1]
    y = data[:,-1]
    
    return X, y

def K(xi, xj, gamma) :
    delta = xi - xj
    sumSquare = float(delta.dot(delta.T))
    result = math.exp( -1 * gamma * sumSquare)
    return result

def O(xi, xj, gamma):
    omega = K(xi, xj, gamma)
    return omega

def train(x, y, C, gamma):
    n = len(y) + 1
    
    A = np.zeros((n, n))
    A[0][0] = 0
    A[0, 1:n] = np.ones((n-1))
    A[1:n, 0] = np.ones((n-1))
    for i in range(1, n):
        for j in range(1,n):
            A[i, j] = O(x[i-1], x[j-1], gamma)
            if(i==j):
                A[i, j] += 1/C
    
    B = np.ones((n, 1))
    B[0][0] = 0
    B[1:n, 0] = y.T
    
    return np.linalg.solve(A, B)

# (wt * phiXi + b ) * yi > 1 - ei == (K(xi, xi) + b) * yi
def get_classifier(x, y, C, gamma):
    parameters = train(x, y, C, gamma)
    n = len(x)
    b = parameters[0]
    a = parameters[1:n+1]
    
    return a, b

def classify_points(a, b, X, gamma):
    n = len(X)
    classified = np.zeros(n)
    for i in range(0, n):
        classified[i] = b
        for j in range(0, n):
            classified[i] += a[j] * K(X[i], X[j], gamma) + (a[j]/gamma)
    return classified

def calc_acc(classified, y) :
    yhat = []
    
    for c in classified:
        val = np.sign(c)
        yhat.append(int(val))

    falseNeg = 0
    falsePos = 0
    trueNeg = 0
    truePos = 0
    for i in range(0, len(y)) :
        my_y = int(y[i])
        
        pred_y = int(yhat[i])
        if(my_y > pred_y) :
            falseNeg = falseNeg + 1
        
        elif (my_y < pred_y) :
            falsePos = falsePos + 1
        
        elif (my_y==pred_y and my_y==1) :
            truePos = truePos + 1
        
        elif (my_y==pred_y and my_y==-1) :
            trueNeg = trueNeg + 1
        
    total = falseNeg + falsePos + trueNeg + truePos
    incorrect = falseNeg + falsePos

    errorRate = incorrect/total

    return errorRate
    
def plot_contours(ax, clf, xx, yy, **params) :
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=('k',), linestyles=('-',), linewidths=(1,))
    out = ax.contourf(xx, yy, Z, **params)
    ax.clabel(cs, fmt = '%2.1d', colors = 'k', fontsize=10)
    return out

def make_meshgrid(x, y, h=.2):
    x_min, x_max = x.min() -1, x.max() +1
    y_min, y_max = y.min() -1, y.max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

################## Handmade model ########################
X,y = init_data('train')
## Testing data
X_test, y_test = init_data('test')
# For testing
Cs = [0.1, 1, 10, 100]
gs = [1, 2, 4, 8]
kernals = ['linear', 'rbf', 'poly', 'sigmoid']
# For Plotting, one at a time so it's not too small
# Cs = [1, 2]
# gs = [0.5, 1, 4, 16]
# Cs = [0.1, 1, 5, 10]
# gs = [1, 2]
# Cs = [0.1, 1]
# gs = [1, 2]
# Cs = [10, 100]
# gs = [4, 8]
own_train_accs = np.zeros((len(Cs), len(gs)))
own_test_accs = np.zeros((len(Cs), len(gs)))
own_train_err = np.zeros((len(Cs) * len(gs)))
own_test_err = np.zeros((len(Cs) * len(gs)))
i = 0
j = 0
k = 0
for C in Cs :
    for g in gs :
        train_a, train_b = get_classifier(X, y, C, g)

        ## Training data
        y_pred_train = classify_points(train_a, train_b, X, g)

        train_error = calc_acc(y_pred_train, y)
        train_acc = 1 - train_error
        own_train_accs[i][j] = train_acc
        own_train_err[k] = train_error
        y_pred_test = classify_points(train_a, train_b, X_test, g)

        test_error = calc_acc(y_pred_test, y_test)
        test_acc = 1 - test_error
        own_test_accs[i][j] = test_acc
        own_test_err[k] = test_error
        j = j + 1
        k = k + 1
    j = 0
    i = i + 1
print("------ Own Train Accuracies C vs. G ---------------")
print(own_train_accs)
fig, ax = plt.subplots(2,1)
ax[0].boxplot(own_train_err)
ax[0].set_title("LS-SVM Train Error")
print("------ Own Test Accuracies C vs. G ---------------")
print(own_test_accs)
ax[1].boxplot(own_test_err)
ax[1].set_title("LS-SVM Test Error")
plt.show()
# ######################## SVM from Scikit ########################
i = 0
j = 0
k = 0
sklearn_train_accs = np.zeros((len(Cs), len(gs)))
sklearn_test_accs = np.zeros((len(Cs), len(gs)))

for C in Cs :
    for g in gs :
        
        svclassifier = SVC(kernel='rbf', C=C, gamma=g)
        svclassifier.fit(X, y)
        y_pred_train_sklearn = svclassifier.predict(X)
        train_error_sklearn = calc_acc(y_pred_train_sklearn, y)
        train_acc = 1 - train_error_sklearn
        sklearn_train_accs[i][j] = train_acc
        y_pred_test_sklearn = svclassifier.predict(X_test)
        test_error_sklearn = calc_acc(y_pred_test_sklearn, y_test)
        test_acc = 1 - test_error_sklearn
        sklearn_test_accs[i][j] = test_acc
        j = j + 1
    j = 0
    i = i + 1
kCs = [0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
sklearn_train_err = {'linear': np.zeros(len(kCs)), 'rbf': np.zeros(len(kCs)), 'poly': np.zeros(len(kCs)), 'sigmoid': np.zeros(len(kCs))} 
sklearn_test_err = {'linear': np.zeros(len(kCs)), 'rbf': np.zeros(len(kCs)), 'poly': np.zeros(len(kCs)), 'sigmoid': np.zeros(len(kCs))}

for kernal in kernals:
    k = 0
    for C in kCs:
        svclassifier = SVC(kernel=kernal, C=C, gamma=1)
        svclassifier.fit(X, y)
        y_pred_train_sklearn = svclassifier.predict(X)
        train_error_sklearn = calc_acc(y_pred_train_sklearn, y)
        y_pred_test_sklearn = svclassifier.predict(X_test)
        test_error_sklearn = calc_acc(y_pred_test_sklearn, y_test)
        sklearn_train_err[kernal][k] = train_error_sklearn
        sklearn_test_err[kernal][k] = test_error_sklearn
        k = k + 1
        
    print(sklearn_train_err[kernal])
    print(sklearn_test_err[kernal])
print("------ SKLearn Train Accuracies C vs. G ---------------")
print(sklearn_train_accs)
fig, ax = plt.subplots(1,4)
for i, kernal in enumerate(kernals):
    ax[i].boxplot(sklearn_train_err[kernal])
    ax[i].set_title("SVM Train Error - " + kernal)
plt.show()
plt.close()
print("------ SKLearn Test Accuracies C vs. G ---------------")
print(sklearn_test_accs)
figs, axs = plt.subplots(1,4)
for i, kernal in enumerate(kernals):
    axs[i].boxplot(sklearn_test_err[kernal])
    axs[i].set_title("SVM Train Error - " + kernal)
plt.show()

############### Plots #################

gamma = 1
C = 1
posSVs = []
negSVs = []
posNSVs = []
negNSVs = []
svclassifier = SVC(kernel='rbf', C=C, gamma=gamma)
svclassifier.fit(X, y)
svIndecies = svclassifier.support_

for i, yi in enumerate(y) : 
    if(yi>0):
        if(i in svIndecies):
            posSVs.append(i)
        else:
            posNSVs.append(i)
    else:
        if(i in svIndecies):
            negSVs.append(i)
        else:
            negNSVs.append(i)
X0, X1 = X[:,0], X[:,1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(12,9))
fig.patch.set_facecolor('white')

ax.scatter(X[posSVs, 0], X[posSVs, 1], alpha=0.9, facecolor="purple", edgecolor="purple", s=50, label=" Positive Support Vectors")
ax.scatter(X[negSVs, 0], X[negSVs, 1], alpha=0.9, facecolor="orange", edgecolor="orange", s=50, label=" Negative Support Vectors")
ax.scatter(X[posNSVs, 0], X[posNSVs, 1], alpha=0.3, facecolor="purple", edgecolor="purple", s=50, label=" Positive Non-Support Vectors")
ax.scatter(X[negNSVs, 0], X[negNSVs, 1], alpha=0.3, facecolor="orange", edgecolor="orange", s=50, label=" Negative Non-Support Vectors")
out = plot_contours(ax, svclassifier, xx, yy, cmap="PuOr", alpha=0.4)
plt.colorbar(out, shrink=0.8, extend='both')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
plt.title("SVM Regression", fontsize=13)
plt.show()

rows = len(gs)
columns = len(Cs)
fig, axs = plt.subplots(rows, columns)

X0, X1 = X[:,0], X[:,1]
xx, yy = make_meshgrid(X0, X1)
for ix, c in enumerate(Cs) :
    for jx, g in enumerate(gs) :
        posSVs = []
        negSVs = []
        posNSVs = []
        negNSVs = []
        svclassifier = SVC(kernel='rbf', C=c, gamma=g)
        svclassifier.fit(X, y)
        svIndecies = svclassifier.support_

        for i, yi in enumerate(y) : 
            if(yi>0):
                if(i in svIndecies):
                    posSVs.append(i)
                else:
                    posNSVs.append(i)
            else:
                if(i in svIndecies):
                    negSVs.append(i)
                else:
                    negNSVs.append(i)
        axs[jx, ix].scatter(X[posSVs, 0], X[posSVs, 1], alpha=0.9, facecolor="purple", edgecolor="purple", s=50, label=" Positive Support Vectors")
        axs[jx, ix].scatter(X[negSVs, 0], X[negSVs, 1], alpha=0.9, facecolor="orange", edgecolor="orange", s=50, label=" Negative Support Vectors")
        axs[jx, ix].scatter(X[posNSVs, 0], X[posNSVs, 1], alpha=0.3, facecolor="purple", edgecolor="purple", s=50, label=" Positive Non-Support Vectors")
        axs[jx, ix].scatter(X[negNSVs, 0], X[negNSVs, 1], alpha=0.3, facecolor="orange", edgecolor="orange", s=50, label=" Negative Non-Support Vectors")
        out = plot_contours(axs[jx, ix], svclassifier, xx, yy, cmap="PuOr", alpha=0.4)
        fig.colorbar(out, shrink=0.8, extend='both', ax=axs[jx, ix])
        axs[jx, ix].set_title("C={}, gamma={}".format(c, g), fontsize=10)
plt.show()

