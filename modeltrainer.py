from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, BayesianRidge 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.feature_selection import RFECV

import numpy as np

import csv
import random
import multiprocessing
import time


def check_combos(features, targets, estimator, start, stop, lock, counter, best_s, best, train_split): 
   """
   Helper function for feature_select. Iterates from the "start"th potential combination of features to the "stop"th combination. Fits estimator with these
   combinations of features and gets the MSE score of its predictions. Iteratively finds the best (lowest) score and returns the integer representation of 
   the best combination of features (See Notes for description of integer representation).
   Parameters:
      features - the 2-D array of features where the rows are samples and the columns are the different features
      targets - the 1-D array of target values wherein targets[i] corresponds to features[i]
      estimator - the estimator object (must be a regressor)
      start - the integer representation of the first combination to try
      stop - the integer representation of the last combination to try (non-inclusive)
      lock - the multiprocessing lock
      counter - the shared counter to track progress of overall task
      best_s - the shared data for the best score
      best - the shared data for the integer representation of the best combination of features
      train_split - the percent of data to be used as training data. In range (0,1)
   Returns:
      No return value. Updates shared/external values counter, best_s, and best
   Notes:
      The combinations are represented as binary strings with length equal to the number of features. A 0 means the feature is not used, and a 1 means that 
      it is used. In order to work with these values easily, they are stored as the integer form of these binary numbers.
      ex.
         With 4 features, we want to represent the combination of the first, second, and fourth features
         '1101' -> 13
   """
   # split training/testing data
   num_train = int(train_split * len(features))
   x_test, x_train, y_test, y_train = features[num_train:], features[:num_train], targets[num_train:], targets[:num_train] 

   # initialize best_score, best_combo to be those which use all of the features
   estimator.fit(x_train, y_train)
   best_score = MSE(y_test, estimator.predict(x_test))
   best_combo = format(start, f'0{len(features[0])}b')

   # increment shared counter variable when the lock is secured
   with lock:
      counter.value += 1

   # iterate over this subprocess's range of combinations of features and test
   for j in range(start + 1, stop):
      # translate from integer feature-combo representation to an array of features
      curr_features = np.array([])
      curr_combo = format(j, f'0{len(features[0])}b') # convert int -> binary string
      # convert binary string -> array of features
      for i in range(len(curr_combo)):
         if curr_combo[i] == '1': 
            if len(curr_features) == 0:
               curr_features = features[:, [i]]
            else:
               curr_features = np.append(curr_features, features[:, [i]], axis=1)

      # test and train model on new feature combo
      x_test, x_train = curr_features[num_train:], curr_features[:num_train]
      estimator.fit(x_train, y_train)
      curr_score = MSE(y_test, estimator.predict(x_test))

      # update best_score and best_combo if this score is better
      if curr_score < best_score:
         best_score = curr_score
         best_combo = curr_combo

      # increment shared counter variable when the lock is secured 
      with lock:
         counter.value += 1
   
   # once lock is secured, check if this process's best score is better than the other finished proccesses'
   with lock:
      if best_score < best_s.value:
         best.value = int(best_combo, 2)



def feature_select(features, targets, estimator, train_split=0.8, num_processors=8):
   """
   Try every combination of features to determine the best using MSE. Only use with few features or else this will take a while
   Params:
      features - a 2-D array of the data of the features to select from
      targets - a 1-D array of the target data
      estimator - the estimator to use in the feature selection
      train_split - the percent of data to be used for training. in range (0,1)
      num_processors - number of processes to use for computation. Must be 1 or more
   Returns:
      A 2-D array of the data best feature combination
   """
   features = np.array(features)

   # calculate necessary values to divide up task between processes
   num_combos = int(np.power(2, len(features[0])) - 1)
   increment = int(num_combos / num_processors)

   # initialize shared variables
   counter = multiprocessing.Value('i', 0) 
   best_s = multiprocessing.Value('f', np.inf)
   best = multiprocessing.Value('i')
   lock = multiprocessing.Lock()
   processes = []

   # create subprocesses
   for i in range(num_processors):
      if i == num_processors - 1: # gives the last process the leftover combos of features as well
         processes.append(multiprocessing.Process(target=check_combos, args=(features, 
                                                targets, estimator, i * increment, num_combos, 
                                                lock, counter, best_s, best, train_split)))
      else:
         processes.append(multiprocessing.Process(target=check_combos, args=(features, 
                                                targets, estimator, i * increment, (i + 1) * increment, 
                                                lock, counter, best_s, best, train_split)))
      processes[i].start()
   
   # every 10 seconds, print progress of job until job is finished or gets stuck for 10 seconds
   prev = 0
   while True:
      with lock:
         print(f'Finished with {counter.value} / {num_combos}')
         if counter.value == num_combos or counter.value == prev:
            break
         prev = counter.value

      time.sleep(10)
      
   # rejoin processes
   for i in range(8):
      processes[i].join()
   
   # store best features as an array
   best_combo = format(best.value, f'0{len(features[0])}b')
   best_features = np.array([])
   for i in range(len(best_combo)):
         if best_combo[i] == '1':
            if len(best_features) == 0:
               best_features = features[:, [i]]
            else:
               best_features = np.append(best_features, features[:, [i]], axis=1)   

   return best_features


if __name__ == '__main__': # this is necessary for the multiprocessing in the feature selection to work
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


   # REGRESSION MODELS
   # Test different regression models and combinations of parameters. This part has been commented out it does not run every time
   # something is to be changed about the model.

   # # LINEAR REGRESSOR - 0.02111595 MSE
   # linear = LinearRegression()
   # linear.fit(x_train, y_train)
   # linear_eval = MSE(y_test, linear.predict(x_test))
   # print(linear_eval)

   # # ELASTIC NET - 1.07 MSE
   # en_params = [
   #    {
   #       'alpha': [0.25, 0.5, 0.75, 1],
   #       'l1_ratio': [0.25, 0.5, 0.75],
   #       'selection': ['cyclic', 'random'],
   #       'max_iter': [10000]
   #    }
   # ]
   # en = ElasticNet()
   # en_search = GridSearchCV(en, en_params, scoring='neg_mean_squared_error').fit(x_train, y_train)
   # en_eval = MSE(y_test, en_search.predict(x_test))
   # print(en_eval) 

   # # STOCHASTIC GRADIENT DESCENT - 0.04066 MSE (much higher normally)
   # sgd_params = [
   #    {
   #       'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
   #       'penalty': ['l2', 'l1', 'elasticnet'],
   #       'alpha': [0.0001],
   #       'epsilon': [0.1],
   #       'learning_rate': ['constant', 'optimal', 'invscaling'],
   #       'max_iter': [10000]
   #    }
   # ]
   # sgd = SGDRegressor()
   # sgd_search = GridSearchCV(sgd, sgd_params, scoring='neg_mean_squared_error').fit(x_train, y_train)
   # sgd_eval = MSE(y_test, sgd_search.predict(x_test))
   # print(sgd_eval)

   # # BAYESIAN RIDGE - 0.02111593 MSE
   # br_params = [
   #    {
   #       'n_iter': [300],
   #       'tol': [0.001],
   #       'alpha_1': [0.0000001],
   #       'alpha_2': [0.000002],
   #       'lambda_1': [0.000002],
   #       'lambda_2': [0.00000005]
   #    }
   # ]
   # br = BayesianRidge()
   # br_search = GridSearchCV(br, br_params, scoring='neg_mean_squared_error').fit(x_train, y_train)
   # br_eval = MSE(y_test, br_search.predict(x_test))
   # print(br_eval)
   # print(br_search.best_params_)

   # # GRADIENT BOOSTING REGRESSOR - 0.629 MSE
   # gb_params = [
   #    {
   #       'loss': ['squared_error', 'huber', 'quantile'],
   #       'learning_rate': [0.1, 0.2],
   #       'n_estimators': [100],
   #       'subsample': [0.5, 1],
   #       'criterion': ['friedman_mse']
   #    }
   # ]
   # gb = GradientBoostingRegressor()
   # gb_search = GridSearchCV(gb, gb_params, scoring='neg_mean_squared_error').fit(x_train, y_train)
   # gb_eval = MSE(y_test, gb_search.predict(x_test))
   # print(gb_eval)

   # # LINEAR SVR - 0.02242 MSE
   # lsvr_params = [
   #    {
   #       'epsilon': [0, 0.5],
   #       'C': [1, 0.5],
   #       'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
   #       'intercept_scaling': [1, 1.5],
   #       'dual': [False]
   #    }
   # ]
   # lsvr = LinearSVR()
   # lsvr_search = GridSearchCV(lsvr, lsvr_params, scoring='neg_mean_squared_error').fit(x_train, y_train)
   # lsvr_eval = MSE(y_test, lsvr_search.predict(x_test))
   # print(lsvr_eval)


   # Bayesian Ridge proved to be marginally better than a standard Linear Regression model, so we will proceed with a Bayesian Ridge
   br = BayesianRidge(n_iter=300, tol=0.001, alpha_1=0.0000001, alpha_2=0.000002, lambda_1=0.000002, lambda_2=0.00000005)

   # Feature Learning
   print(x[0]) # all features
   br.fit(x_train, y_train)
   print(MSE(y_test, br.predict(x_test))) # 0.021012464747050024 MSE
   x = feature_select(x, y, br)
   print(x[0]) # Age, Wins, Losses, Minutes Played, Points, Field Goals Made, 3 Pointers Made, Free Throws Made, Free Throws Attempted, OREB, DREB, Assists, TurnOVs, Steals, Blocks, +/-
   x_train, x_test = x[:num_train], x[num_train:]
   br.fit(x_train, y_train)
   print(MSE(y_test, br.predict(x_test))) # 0.021005883105879752 MSE

   # So we drop Field Goals Attempted, 3 Pointers Attempted, and Personal Fouls


