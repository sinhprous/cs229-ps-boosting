import numpy as np

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def find_best_threshold(X, y, p_dist):
    flipped = 1
    m = X.shape[0]
    n = X.shape[1]
    ind = 0
    thresh = 0
    min = np.inf

    for j in range(n):
        X_sort = np.sort(X[:, j])[::-1] # desc
        argsort = np.argsort(X[:, j])[::-1]
        y_sort = y[argsort]
        p_sort = p_dist[argsort]
        possible_thresh = [X_sort[0]+1]
        possible_thresh.extend([(X_sort[i] + X_sort[i + 1]) / 2 for i in range(len(X_sort) - 1)])
        obj1 = np.array(p_dist).dot(np.array(y_sort==1))
        for i in range(len(possible_thresh)):
            if i != 0:
                obj1 -= p_sort[i]*y_sort[i]
            obj = 0.0
            obj2 = 1 - obj1
            if obj1 < obj2:
                obj = obj1
                flipped = 1
            else:
                obj = obj2
                flipped = -1

            if obj < min:
                min = obj
                ind = j
                thresh = possible_thresh[i]

    return ind, thresh, flipped