import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

tmp = sio.loadmat("mousetracks.mat")

tracks = {}
for trackno in range(30):
    tracks[trackno] = tmp["num%d"%(trackno)]
    
plt.close("all")
for trackno in range(30):
    plt.plot(tracks[(trackno)][:,0],tracks[(trackno)][:,1],'.')
plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()

X = np.zeros([30*50,2])

for trackno in range(30):
    X[(trackno*50):((trackno+1)*50),:] = tracks[trackno]
    
plt.close("all")
plt.plot(X[:,0],X[:,1],'.')

plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()

def kmeans(X,K=5,maxiter=100):
    
    clusters = np.random.rand(K,2)
    max_x = np.max(X[:,0])
    min_x = np.min(X[:,0])
    range_x = max_x - min_x
    max_y = np.max(X[:,1])
    min_y = np.min(X[:,1])
    range_y = max_y - min_y
    clusters[:,0] = clusters[:,0] * range_x + min_x
    clusters[:,1] = clusters[:,1] * range_y + min_y
    assgn = []
    for k in range(K):
        assgn.append([])
    mindist=1000
    for iter in range(maxiter):
        # cluster assignment update
        for x in X:
            xx = x[0]
            xy = x[1]
            mindist = 1000
            for k in range(K):
                kx = clusters[k][0]
                ky = clusters[k][1]
                dist = np.sqrt((kx - xx)**2 + (ky-xy)**2)
                if(dist<mindist):
                    mindist = dist
                    mink = k
            assgn[mink].append(x) 
        
        for k in range(K):
            # cluster center update
            mean = [0,0]
            for a in assgn[k]:
                mean[0] = mean[0] + a[0]
                mean[1] = mean[1] + a[1]
            sz = len(assgn[k])
            if(sz == 0):
                sz = 1
            mean[0] = mean[0]/sz
            mean[1] = mean[1]/sz
            clusters[k][ 0] = mean[0]
            clusters[k][1] = mean[1]
            
    return clusters
            
kmeancluster = kmeans(X)
plt.close("all")
plt.plot(X[:,0],X[:,1],'b.')
plt.plot(kmeancluster[:,0],kmeancluster[:,1],'ro')
#uncomment to plot your cluster centers
#plt.plot(C[:,0],C[:,1],'ro')
plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()


