#!/usr/bin/env python
# coding: utf-8

# Helper Funktionen

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import math
from sklearn.metrics import mean_squared_error


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
#target = 'SalePrice'

def checkFeature(feature, data, is_cat):
    checkNAs(feature, data)
    checkForNegatives(feature, data)
    overview(feature, data)
    plotDistribution(feature, data, is_cat)
    if feature != target:
        plotRelationToTarget(feature, data)

def checkForNA(feature, data):
    if data[feature].isna().sum() > 0:
        return True
    else:
        return False

def checkFeatureNAs(feature, data):
    if data[feature].isna().sum() > 0:
        print(bcolors.FAIL + "Sum NAs: " + str(data[feature].isna().sum()))
    else:
        print(bcolors.OKGREEN + "No NAs" +bcolors.ENDC)
    
def checkDatasetNAs(data):
    sum_cols = len(data.columns)
    for col in data.columns: 
        if not checkForNA(col, data):
            sum_cols -= 1

    if sum_cols == 0:
        print(bcolors.OKGREEN + "No NAs in Dataset" +bcolors.ENDC)
    
def checkForNegatives(feature, data):
    if any(data[feature]<0):
        print (bcolors.WARNING + "Warning feature has negative value!" + bcolors.ENDC)
    else:
        print (bcolors.OKGREEN + "No negative values" + bcolors.ENDC)

def plotDistribution(feature, data, is_cat):
    sns.distplot(data[feature], fit=norm);
    fig = plt.figure()
    if not is_cat:
        res = stats.probplot(data[feature], plot=plt)
    
def plotRelationToTarget(feature, data):
    data_temp = pd.concat([data[target], data[feature]], axis=1)
    data_temp.plot.scatter(x=feature, y=target, ylim=(0,800000));
        
def overview(feature, data):
    print(data[feature].describe())
    print(bcolors.HEADER + "Head" +bcolors.ENDC)
    print(data[feature].head(3))
    
def printSkewKurt(feature, data):
    print("Skewness: %f" % data[feature].skew())
    print("Kurtosis: %f" % data[feature].kurt())
    
def calculate_performance(prediction, actual, scaler):
    if scaler == True:
        p = scaler.inverse_transform(prediction.reshape(-1,1))
        a = scaler.inverse_transform(actual.reshape(-1,1))
    else:
        p = prediction
        a = actual
        
    mse = mean_squared_error(a, p)
    err = np.sqrt(mse)
    r2 = r2_score(a, p)
    mae = median_absolute_error(a, p)
    
    return (mse, err, r2, mae)

def print_performance(measure_tuple):
    
    mse = measure_tuple[0]
    err = measure_tuple[1]
    r2 = measure_tuple[2]
    mae = measure_tuple[3]
    
    print("Mean squared error is {}".format(str(mse)))
    print("Positive mean error is {}".format(str(err)))
    print("Overall R² is {}".format(str(r2)))
    print("Median absolute error is {}".format(str(mae)))

def eval_model(model, test_X, test_y):
    r2 = model.score(test_X, test_y)

    pred_y = model.predict(test_X)
    rmse = math.sqrt(mean_squared_error(np.exp(test_y), np.exp(pred_y)))
    
    print('r2 = ' + str(r2))
    print('rmse = ' + str(rmse))
    return rmse, r2

def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        return True 
    else: 
        return False


from matplotlib import pyplot as plt
from keras.models import Model

def plot_history(history, save=False):
    """This method plots the learning history of a ANN from a history object.

    There will be two plots, a plot of the loss function and a plot of the
    accuracy function. This requires that accuracy is recorded during
    execution of the fit method.

    Keyword arguments:
        - history: A tensorflow history object
    
    Returns:
        - None
    
    Raise:
        - KeyError if "acc" missing in history.keys()
        - KeyError if "loss" missing in history.keys()
    """

    if "acc" not in history.history.keys():
        raise KeyError("Accuracy missing in history. Please record!")
    if "loss" not in history.history.keys():
        raise KeyError("Loss missing in history. Please record!")

    plt.figure(figsize=(20,6))
    
    p1 = plt.subplot2grid((1,2), (0,0))
    p2 = plt.subplot2grid((1,2), (0,1))

    p1.plot(history.history['loss'], label="train")
    p1.plot(history.history['val_loss'], label="test")
    p1.set_xlabel('epochs')
    p1.set_ylabel('loss')
    p1.set_title('Loss development')
    p1.legend(loc=0)
    
    
    p2.plot(history.history['acc'], label="train")
    p2.plot(history.history['val_acc'], label="test")
    p2.set_xlabel('epochs')
    p2.set_ylabel('accuracy')
    p2.set_title('Accuracy development')
    p2.legend(loc=0)
    
    if save:
        plt.savefig('history.png')
   
    plt.show()