import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def read_data():
    data = []
    with open('transfusion.csv') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader) # take the header out
        for row in reader: # each row is a list
            data.append(row)
    data  = np.array(data, dtype = np.int32)
    np.random.shuffle(data)
    X = data[:,:-1]
    y = data[:,-1]
    

    return X, y

    
X, y = read_data()

fold_size = int(np.round(len(X)/5))
folds = [X[0:fold_size], X[fold_size:(2 * fold_size)], X[(2 * fold_size):(3 * fold_size)], X[(3 * fold_size):(4 * fold_size)], X[(4*fold_size):(len(X))]]
folds_truth = [y[0:fold_size], y[fold_size:(2 * fold_size)], y[(2 * fold_size):(3 * fold_size)], y[(3 * fold_size):(4 * fold_size)], y[(4*fold_size):(len(y))]]
c_values = [.1, 1, 10, 100]
c_avgs = [0, 0, 0, 0]
tests = []
for i in range(0,len(folds)):
    test_fold = folds[i]
    batch_c_val = -100
    batch_c = -100
    for j in range(0,len(folds)):
        validation_fold = folds[j]
        validation_truth = folds_truth[j]
        training_set = np.zeros((0, 4))
        training_truth = np.zeros((0))
        if(i == j):
            continue
        for k in range(0,len(folds)):
            training_fold = folds[k]
            if((k == j) | (k==i)):
                continue
            training_set = np.concatenate((training_set, training_fold))
            training_truth = np.concatenate((training_truth,folds_truth[k]))
        for C in c_values:
            c = c_values.index(C)
            model = LogisticRegression(C=C)
            model = model.fit(training_set, training_truth) # training
            y_pred = model.predict(validation_fold) # predicting
            score = float(f1_score(validation_truth, y_pred))
            c_avgs[c] = c_avgs[c] + score
            print('Training f1-score for C = %f:' % C, score)
        c_avgs[:] = [x / 4 for x in c_avgs]

        best_c_val = np.max(c_avgs)
        c_i = c_avgs.index(best_c_val)
        best_c = c_values[c_i]
        print('Best C for Test Cycle %d was %f with a score of %f' % (i, best_c, best_c_val))
        if(best_c_val > batch_c_val):
            batch_c_val = best_c_val
            batch_c = best_c

    test_training_data = np.zeros((0, 4))
    test_training_truth = np.zeros((0))
    _training_data = [x for l,x in enumerate(folds) if l!=i]
    for data in _training_data:
        test_training_data = np.concatenate((test_training_data, data))
    _training_truth = [x for l,x in enumerate(folds_truth) if l!=i]
    for data in _training_truth:
        test_training_truth = np.concatenate((test_training_truth, data))
    model = LogisticRegression(C=batch_c)
    model.fit(test_training_data, test_training_truth) 
    test_results = model.predict(test_fold)
    score = float(f1_score(test_results, folds_truth[i]))
    print("Test score for fold %d : %f" % (i, score))
    tests.append(score)

std = np.std(tests)
mean = np.mean(tests)

for test in tests :
    
    print("Test Score: \n", str(test))

print("Std Dev: ", std)
print("Mean: ", mean)


