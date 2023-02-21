# This is the condensed version of modeltrainer.py; does not derive best hyperparameters, etc.
# Author: Ethan Sims

from sklearn.linear_model import BayesianRidge 
from sklearn.metrics import mean_squared_error as MSE

import numpy as np

import csv
import random
import multiprocessing
import time

# read in data
data_reader = csv.reader(open('resources\player_stats.csv', 'r'))

x = []
y = []

for row in data_reader:
   if len(row) == 0 or row[0] == 'AGE': # ignore empty rows and first row
      continue
   else:
      for i in range(len(row)):
         row[i] = float(row[i])
      y.append(row.pop(18))
      # remove features not included in optimal feature combination (see modeltrainer.py)
      row.pop(6)
      row.pop(8)
      row.pop(17)

      x.append(row)

# shuffle data
temp = list(zip(x, y))
random.shuffle(temp)
x, y = zip(*temp)
x, y = list(x), list(y)

# split data
num_train = int(len(x) * 0.8)
x_train, x_test = x[:num_train], x[num_train:]
y_train, y_test = y[:num_train], y[num_train:]


# Train model
br = BayesianRidge(n_iter=300, tol=0.001, alpha_1=0.0000001, alpha_2=0.000002, lambda_1=0.000002, lambda_2=0.00000005)
br.fit(x_train, y_train)
print(MSE(y_test, br.predict(x_test)))
