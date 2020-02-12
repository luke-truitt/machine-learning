import numpy as np
import matplotlib.pyplot as plt
x1 = [1, 1]
x2 = [0, -1]
x3 = [-2, 1]
w = [0,0]
y = [1, -1, 1]

y_clr = {1 : 'b', -1 : 'r'}
## Original Data
# a = -1
# b = 2
# c = 0

## First change
# a = 1
# b = 3
# c = 0
# x1 = [1, 0]

## Second change
# a = -5
# b = 6
# c = 0
# x3 = [-10, 1]

x = [x1, x2, x3]

def mag(pt) :
    curr = np.sqrt(pt[0] * pt[0] + pt[1] * pt[1])
    return curr

def norm(xarr) :
    mx = 0
    for xpt in xarr :
        curr = mag(xpt)
        if curr > mx:
            mx = curr
    
    xrtn = []
    for xpt in xarr :
        xrtn.append(xpt/mx)
    return xrtn
    
def dist(xarr) :
    for xpt in xarr :
        numerator = np.abs(w[0] * xpt[0] + w[1] * xpt[1])
        denomonator = np.sqrt(w[0] * w[0] + w[1] * w[1])

        print(numerator/denomonator)

def train() :
    count = 0 
    i = 0
    iters = 0
    while(count < len(x)) :
        iters = iters + 1
        if i >= len(x) :
            i = 0
        x_curr = x[i]
        sm = 0
        j = 0
        for j in range(len(x_curr)):
            sm = sm + (x_curr[j] * w[j])
            j = j + 1
        j = 0
        if(((sm > 0) & (y[i] > 0)) | ((sm < 0) & (y[i]<0))):
            count = count + 1
            i = i + 1
            print("Hit")
            continue
        else:
            count = 0
            for j in range(len(x_curr)):
                w[j] = w[j] + (y[i] * x_curr[j])
            i = i + 1
            print("Miss")
        print(w)
    print(str(w) + " Got it!")
    return w
    
      
def plot(xarr) :
    i = 0
    for pt in xarr :
        plt.plot(pt[0], pt[1], y_clr[y[i]] + 'o')
        i = i+1
    x1 = 1
    pt1 = (-1 * (w[0]*x1))/w[1]
    x2 = -1
    pt2 = (-1 * (w[0]*x2))/w[1]
    plt.plot([x1, x2], [pt1, pt2])
    plt.axis([-1, 1, -1, 1])
    plt.show()

train()
x = norm(x)

dist(x)

plot(x)