import numpy as np
from find_best_threshold import find_best_threshold, sign

def stump_booster(X, y, T):
    m = X.shape[0]
    n = X.shape[1]

    p_dist = np.array([1.0/m]*m)

    theta = []
    feature_inds = []
    thresholds = []
    flippeds = []

    for iter in range(T):
        if iter%10==0:
            print("iter: %d"%iter)
        ind, thresh, flipped = find_best_threshold(X, y, p_dist)
        feature_inds.append(ind)
        thresholds.append(thresh)
        flippeds.append(flipped)

        if (iter > 0):
            p_dist = np.array([np.exp(-y[i]*np.array(theta).dot(np.array([sign(X[i][feature_inds[j]]-thresholds[j]) for j in range(iter)]))) for i in range(m)])
            p_dist = p_dist/np.sum(p_dist)

        W_pos = np.sum([p_dist[i] for i in range(len(p_dist)) if y[i]*sign(X[i][ind]-thresh) == 1])
        W_neg = np.sum([p_dist[i] for i in range(len(p_dist)) if y[i]*sign(X[i][ind]-thresh) == -1])
        if W_neg != 0 and W_neg != 0:
            newest_theta = 1/2.0*np.log(W_pos/W_neg)
        else:
            newest_theta = 0.0
        theta.append(newest_theta)

    return np.array(theta), np.array(feature_inds), np.array(thresholds), np.array(flippeds)