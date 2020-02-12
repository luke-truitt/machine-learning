import numpy as np
import matplotlib.pyplot as plt
import csv;
tree = {}
def read_data():
    data = {}
    with open('decisiontree.csv') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader) # take the header out
        for head in headers:
            if(head=="f"):
                continue
            data.update({head: []})
        for row in reader: # each row is a list
            y = row[len(row)-1]
            for i in range(0, len(row)-1):
                data[headers[i]].append([row[i], y])
    return data

def build_tree() :
    data = read_data()

    print(min_entropy(data))


def min_entropy(data):
    min_e = 100000000
    min_f = ''
    min_splits = {}
    ## Iterate through each feature
    for feature in data:
        ## Check what the entropy would be based on choosing "feature"
        feature_data = data[feature] # Array of features and outputs
        splits = calc_splits(feature_data)
        entropy = calc_entropy(splits, feature_data)
        if(entropy < min_e):
            min_e = entropy
            min_f = feature
            min_splits = splits

    return min_f + ": " + str(min_e) + "\n" + str(min_splits)
        

            
def calc_splits(feature_data):
    splits = {"0": [], "1": []}
    for point in feature_data:
        if(point[0]==0):
            splits["0"].append(point)
        else:
            splits["1"].append(point)
    return splits        

def calc_entropy(splits, feature_data):

    total = len(feature_data)
    ## Breakdowns of data for each direction (zero or one) moved on the tree
    zpos = 0
    zneg = 0
    opos = 0
    oneg = 0
    ## Sort the data
    for data in splits["0"]:
        if(data[1]==0):
            zneg = zneg + 1
        else:
            zpos = zpos + 1
    for data in splits["1"]:
        if(data[1]==0):
            oneg = oneg + 1
        else:
            opos = opos + 1
    ## Calculate overall values
    pos = zpos + opos
    neg = zneg + oneg
    ## Calculate the probabilities for the whole level
    tp_pos = pos/total
    tp_neg = neg/total
    ## Calculate current level entropy
    current_entropy = entropy(tp_pos, tp_neg)

    sub_entropy = sub_entropy_calc(opos, oneg, zpos, zneg)

    entropy_result = current_entropy - sub_entropy

    return entropy_result    

def entropy(p_pos, p_neg) :
    
    if((p_pos == 0) | (p_neg == 0)):
        return 0
    h_pos = (-1 * p_pos * np.log2(p_pos))
    h_neg = (-1 * p_neg * np.log2(p_neg))

    return h_pos + h_neg

def sub_entropy_calc(opos, oneg, zpos, zneg):
    total = opos + oneg + zpos + zneg
    if(zpos+zneg==0):
        l_weight = 0
        left_entropy = 0
    else:
        ## Calculate the probablilities for the left side
        l_weight = (zpos + zneg)/total
        lp_neg = zneg/(zpos+zneg)
        lp_pos = zpos/(zpos+zneg)
        ## Calculate the entropy for the left side
        left_entropy = entropy(lp_pos, lp_neg)
    if(opos + oneg==0):
        r_weight = 0
        right_entropy = 0
    else:
        ## Calculate the probablilities for the right side
        r_weight = (opos + oneg)/total
        rp_neg = oneg/(opos+oneg)
        rp_pos = opos/(opos+oneg)
        ## Calculate the entropy for the right side
        right_entropy = entropy(rp_pos, rp_neg)

    sub_entropy = (l_weight * left_entropy) + (r_weight * right_entropy)

    return sub_entropy


e = entropy(.5, .5)
se = sub_entropy_calc(3, 1, 1, 3)
g = e - se
print(str(g))