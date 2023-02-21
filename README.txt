Author: Ethan Sims, ea.sims@me.com

This project was made to predict average fantasy points per game given an NBA player's average statistics per game.
I took a very over-engineered approach considering that many of these statistics directly used in an equation to 
calculate fantasy points, however my approach does allow for the inclusion of factors such as age in the prediction.
In a practical setting, I would likely not waste so much effort on a problem like this, but it provided a good chance
to learn more about machine learning (and multiprocessing/multithreading).

datascraper.py is ran first to get the data. Output of datascraper.py is in resources/player_stats.csv

modeltrainer.py is ran to get the optimal ML model, hyperparameters, and features. 

modeltrainer_condensed.py trains a model based on the optimal model, parameters, etc derived in modeltrainer.py