from scipy.io import loadmat
import numpy as np 
from sklearn.utils import resample
import matplotlib.pyplot as plt

def init_data():

    data = loadmat('mnist_all.mat')
    zero_train  = data['train0']
    zero_test = data['test0']
    one_train = data['train1']
    one_test = data['test1']

    return zero_train, zero_test, one_train, one_test

class net:
    def initialize(self, d, h):
        self.W = []
        self.b = []
        self.d = d
        self.h = h

        self.set_bias(0.01)
        self.set_weights(0.01)

    def set_weights(self, scale):
        for i in range(self.d):
            if i == 0:
                w = scale*np.random.randn(self.h,784)
            else:
                w = scale*np.random.randn(self.h,self.h)
                
            self.W.append(w)

        self.W.append(scale*np.random.randn(self.h))

    def set_bias(self, scale):
        for i in range(self.d):
            b = scale*np.random.randn(self.h)
            self.b.append(b)
            
        self.b.append(0.01*np.random.randn())

    def __init__(self,d,h):
        self.initialize(d,h)
        
    def activate(self,z):
        return 1/(1+np.exp(-z))
    
    def get_zs(self, y, fx, example):
        z_vec = []
        ## Output layer
        z_d = -(y/fx +(y-1)/(1-fx))*fx*(1-fx) 
        z_vec.append(z_d)
        for i in range(self.d, 1, -1):
            ## If we're on the last one match dimension correctly
            if i == self.d:
                z = z_d*self.W[i] * example[i-1]*(1-example[i-1])
            ## Otherwise, want to get vector from gradient_d and W[i]
            else:
                z = np.dot(z_d,self.W[i]) * example[i-1]*(1-example[i-1])
            
            z_vec.append(z)
            z_d = z
        z_vec.append(np.dot(z_d,self.W[1])* example[0]*(1-example[0]))

        return z_vec

    def calc_grads(self, z_vec, output, example):
        dwl = []
        ## Snag the first gradient
        dwd = z_vec[0] * output[self.d-1]
        dwl.append(dwd) 
        
        ## Gradients for each layer: h * z
        for i in range(1,len(z_vec)):    
            if i < len(z_vec) - 1:
                ## 
                dwi = np.array([output[i-1],]*4)
                for j in range(z_vec[i].shape[0]):
                    dwi[j] = z_vec[i][j] * dwi[j]
            else:
                dwi = np.array([example[:-1],]*4)
                for k in range(z_vec[i].shape[0]):
                    dwi[k] =  z_vec[i][k]*dwi[k]
            dwl.append(dwi)
        
        return dwl
    
    def forward_prop(self, x):
        h = x
        h_vec = []
        for i in range(len(self.W)):
            h =  np.matmul(self.W[i],h) + self.b[i] 
            if(i != len(self.W) - 1):
                for j, hj in enumerate(h):
                    h[j] = self.activate(hj)
            else:
                h =self.activate(h)    
            h_vec.append(h)
        return h_vec
    
    def back_prop(self, data, verbose=False):
        dw_vec = []
        db_vec = []

        for i in range(len(self.W)):
            d_w = np.zeros_like(self.W[i])
            dw_vec.append(d_w)
            d_b = np.zeros_like(self.b[i])
            db_vec.append(d_b)

        for example in data:
            out = self.forward_prop(example[:-1])
            n = len(out)
            fx = out[n-1]
            y = example[-1]
            ## Get the z of every layer

            z_vec = self.get_zs(y, fx, out)
            
            ## Layer wise grad wrt weight     
            dwl = self.calc_grads(z_vec, out, example)
            ## Layer wise grad wrt biases are just zl
            dbl = z_vec
                
            for i in range(len(dwl)):
                dw_vec[i] = dw_vec[i]+dwl[-(i+1)]
                db_vec[i] = db_vec[i]+dbl[-(i+1)]
        
        for i in range(len(dwl)):
            dw_vec[i] = dw_vec[i]/len(data)
            db_vec[i] = db_vec[i]/len(data)
        return dw_vec, db_vec
                
    def calc_loss(self,X, verbose=False):
        loss = 0
        for i in X:
            training_features = i[:-1]
            y = i[-1]
            fx = self.forward_prop(training_features)[-1]
            loss = loss - (y*np.log(fx) + (1-y)*np.log(1-fx))
        if(verbose):
            print("Loss this iteration:", loss)
        return loss

    def train(self, X, max_iter = 2000, precision = 400, learning_rate = 0.01, batch_size = 100, verbose=False):
            loss = 1000
            it = 0
            while loss > precision:
                if(verbose):
                    print("Iteration:",it+1)
                batch = resample(X,n_samples = batch_size)
                D_w, D_b = self.back_prop(batch, verbose)
                for i in range(len(self.W)):
                    self.W[i] = self.W[i] - learning_rate *D_w[i]
                    self.b[i] = self.b[i] -  learning_rate *D_b[i]
                loss = self.calc_loss(X, verbose)
                it = it + 1 
                
    def predict(self,X):
        y = []
        for x in X:
            h = self.forward(x)
            if h[self.d] > 0.5:
                y.append(1)
            else:
                y.append(0)
        return y

def get_train_zero_one():
    zero_train_data, zero_test_data, one_train_data, one_test_data = init_data()

    one_train_label = np.ones((one_train_data.shape[0],1))
    zero_train_label = np.zeros((zero_train_data.shape[0],1))
    one_train = np.hstack((one_train_data,one_train_label))
    zero_train = np.hstack((zero_train_data, zero_train_label))
    train = np.vstack((one_train,zero_train))
    np.random.shuffle(train)

    test = {}

    test['0'] = zero_test_data
    test['1'] = one_test_data

    return train, test

def get_acc(pred, num):

    count = pred.shape[0]
    target = np.zeros(count)
    target.fill(num)

    res = pred - target

    correct = 0
    fp = 0
    fn = 0

    for r in res:
        if(r==0):
            correct = correct + 1
        elif(r>0):
            fp = fp + 1
        else:
            fn = fn + 1
    
    acc = correct / count

    err = 1 - acc

    return correct, fp, fn, acc, err




def test_net_zero_one(nn, train_data, test_data):
    zero_train, _, one_train, _ = init_data()
    zero_test = test_data['0']
    one_test = test_data['1']
    test_pred_0 = nn.predict(zero_test)
    _, _, _, test_acc_0, test_err_0 = get_acc(test_pred_0, 0)
    print("Test Accuracy on 0: ", test_acc_0)

    test_pred_1 = nn.predict(one_test)
    _, _, _, test_acc_1, test_err_1 = get_acc(test_pred_0, 0)
    print("Test Accuracy on 1: ", test_acc_1)

    train_pred_0 = nn.predict(zero_train)
    _, _, _, train_acc_0, train_err_0 = get_acc(test_pred_0, 0)
    print("Train Accuracy on 0: ", train_acc_0)

    train_pred_1 = nn.predict(one_train)
    _, _, _, train_acc_1, train_err_1 = get_acc(test_pred_0, 0)
    print("Train Accuracy on 1: ", train_acc_1)

# @param d: number of layers
def train_net(d = 2):

    train_data, test_data = get_train_zero_one()

    # Create Model
    nn = net(d, 4)

    nn.train(train_data, verbose=True)

    test_net_zero_one(nn, train_data, test_data)

    return nn
    
def gradient_mag_calc(d, data):

    gs = np.zeros((d,))
    batch = resample(data,n_samples=200)
    for d in range(1, d+1):

        nn = net(d, 4)
        grad_w, _ = nn.backpropagate(batch)

        gs[d-1] = np.linalg.norm(grad_w[0])

    x = range(1,d+1)
    print(gs)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line, = ax.plot(x, gs, color='blue', lw=2)
    ax.set_yscale('log')
    plt.title('Gradient Magnitude vs. Network Layers')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Gradient Magnitude')
    plt.show()

_ = train_net()
train_data, _ = get_train_zero_one()
gradient_mag_calc(10, train_data)
