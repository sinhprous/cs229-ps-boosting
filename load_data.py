import random_booster
import stump_booster
from find_best_threshold import sign
import numpy as np

X_trains = []
y_trains = []
lines = open("boost_data\\boost_data\\boosting-train.csv").readlines()
for line in lines:
    y_trains.append(int(line.split(",")[0]))
    X_trains.append(list(map(float, line.split(",")[1:])))
X_trains = np.array(X_trains)
y_trains = np.array(y_trains)

X_tests = []
y_tests = []
lines = open("boost_data\\boost_data\\boosting-test.csv").readlines()
for line in lines:
    y_tests.append(int(line.split(",")[0]))
    X_tests.append(list(map(float, line.split(",")[1:])))
X_tests = np.array(X_tests)
y_tests = np.array(y_tests)

'''
theta, ind, thresh, flippeds = stump_booster.stump_booster(X_trains, y_trains, 200)
y_predict = np.array([sign(theta.dot(np.array([sign(x[ind[i]]-thresh[i]) for i in range(len(theta))]))) for x in X_tests])
print(theta)
print(y_predict)
print("stump accuracy = %f"%np.mean(y_predict==y_tests))
'''

theta, ind, thresh = random_booster.stump_booster(X_trains, y_trains, 300)
y_predict = np.array([sign(theta.dot(np.array([sign(x[ind[i]]-thresh[i]) for i in range(len(theta))]))) for x in X_tests])
print("random accuracy = %f"%np.mean(y_predict==y_tests))