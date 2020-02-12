import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import sklearn
from sklearn.datasets import fetch_california_housing

plt.rcParams['font.size'] = 14

tmp = sklearn.datasets.fetch_california_housing()

num_samples   = tmp['data'].shape[0]
feature_names = tmp['feature_names']
y = tmp['target']
X = tmp['data']

data = {}
for n, feature in enumerate(feature_names):
    data[feature] = tmp['data'][:,n]


# Create stumps

# bin the data by proportion, 10% in each bin
bins = {}
bin_idx = (np.arange(0,1.1,0.1)*num_samples).astype(np.int16)
bin_idx[-1] = bin_idx[-1]-1
for feature in (feature_names):
    bins[feature] = np.sort(data[feature])[bin_idx]
    print(bins[feature])

# decision stumps as weak classifiers
# 0 if not in bin, 1 if in bin
stumps = {}
for feature in feature_names:
    stumps[feature] = np.zeros([num_samples,len(bins[feature])-1])
    for n in range(len(bins[feature])-1):
        stumps[feature][:,n] = data[feature]<bins[feature][n+1]
print(stumps)

# stack the weak classifiers into a matrix
H = np.hstack([stumps[feature] for feature in feature_names])
H = np.hstack([np.ones([num_samples,1]),H])
print(H.shape)

# prepare the vector for storing weights
alphas = np.zeros(H.shape[1])
print(alphas.shape)

num_iterations = 30
MSE = np.zeros(num_iterations) # track mean square error


for iteration in range(num_iterations):    
    f = np.matmul(H, alphas) # the current f(x)
    r = y-f; MSE[iteration] = np.mean(r**2) # r = residual
    argmax_func = np.absolute(np.matmul(r, H))
    idx = np.argmax(argmax_func)
    alphas[idx] = alphas[idx] + (np.dot(H[:,idx], r)/np.dot(H[:,idx], np.transpose(H[:,idx]))) # amount to move in optimal direction

alphasf = {}
start = 1
for feature in feature_names:
    alphasf[feature] = alphas[start:(start+stumps[feature].shape[1])]
    start = start + stumps[feature].shape[1]
alphasf['mean'] = alphas[0]

for index, feature in enumerate(feature_names):
    plt.close("all")
    plt.plot(data[feature],y,'.',alpha=0.5,color=[0.9,0.9,0.9])
    # plot stuff
    plt.title(feature)
    plt.xlim([bins[feature][0],bins[feature][-2]])
    plt.xlabel(feature)
    plt.ylabel('House Price')
    plt.show()

for index, feature in enumerate(feature_names):
    plt.close("all")
    plt.plot(data[feature],y-np.mean(y),'.',alpha=0.5,color=[0.9,0.9,0.9])
    # plot stuff
    x_vals = bins[feature]
    y_vals = np.cumsum(alphasf[feature])
    y_vals = np.insert(y_vals, 0, y_vals[0])
    plt.plot(x_vals, y_vals, '-')
    plt.title(feature)
    plt.xlim([bins[feature][0],bins[feature][-2]])
    plt.xlabel(feature)
    plt.ylabel('contribution to house price')
    plt.show()



# variable importance
print("Variable Importance: ")
f = np.matmul(H, alphas)
for i, feature in enumerate(feature_names):
    f_t = np.random.permutation(data[feature])
    H_t = np.copy(H)
    stump = np.zeros([num_samples,len(bins[feature])-1])
    for n in range(len(bins[feature])-1):
        stump[:,n] = f_t>bins[feature][n]
    H_t[:, i*10+1: i*10+11] = stump
    f_t = np.matmul(H_t, alphas)
    r_t = y-f_t
    mse_t = np.mean(r_t**2)
    print(feature, mse_t-MSE[-1])






# ### Boosted Decision Trees


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence



clf = GradientBoostingRegressor(loss="ls")
clf.fit(X,y)



plt.close("all")
plt.figure(figsize=[10,10])
ax = plt.gca()
plot_partial_dependence(clf, X, feature_names, feature_names, n_cols=3, ax=ax) 
plt.tight_layout()
plt.show()


# ### Linear Regression


from sklearn.linear_model import LinearRegression



clf2 = LinearRegression()
clf2.fit(X,y)


# #### Comparison in MSE


print('Lin Reg MSE', np.mean((y-clf2.predict(X))**2))



print('Boosted MSE', np.mean((y-clf.predict(X))**2))
