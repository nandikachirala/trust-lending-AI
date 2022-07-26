from otree.api import *
#from .__init__ import *

import numpy as np
import pandas as pd
from pytest import console_main
import seaborn as sns
from matplotlib import pyplot
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import \
    r2_score, get_scorer
from sklearn.preprocessing import \
    StandardScaler, PolynomialFeatures

import warnings
warnings.filterwarnings('ignore')

def bestDecision(promise5, promise10):
    url = "https://raw.githubusercontent.com/nandikachirala/trust-lending-AI/main/ArtificialData-For-Algorithm.csv"
    dataset = pd.read_csv(url)

    # summarize shape + first few lines
    print(dataset.shape)
    print(dataset.head(10))
    dataset.loc[dataset['AmountSent'] == 5]

    #forming two models
    datasetFive = dataset.loc[dataset['AmountSent'] == 5]
    datasetTen = dataset.loc[dataset['AmountSent'] == 10]
    dataFive = datasetFive.values
    dataTen = datasetTen.values

    # Model for 5

    Xfive, yfive = dataFive[:, :-1], dataFive[:, -1]

    model = Lasso(alpha=1.0)
    # define model evaluation method
    cvFive = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, Xfive, yfive, scoring='r2', cv=cvFive, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    print('Mean Absolute Error: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    # tune alpha for 5
    from sklearn.linear_model import LassoCV
    modelFive = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cvFive, n_jobs=-1)
    modelFive.fit(Xfive, yfive)
    print('alpha: %f' % modelFive.alpha_)

    # predict using tuned alpha
    modelFive = Lasso(alpha=0.18)
    # fit model
    modelFive.fit(Xfive, yfive)
    # new fake data
    row = [promise5, promise10, 5, 1, 1]
    # make a prediction
    predictFive = modelFive.predict([row])
    # summarize prediction
    print('Predicted: %.3f' % predictFive)
    print(modelFive.coef_)
    print(modelFive.intercept_)


    # Model for 10

    Xten, yten = dataTen[:, :-1], dataTen[:, -1]

    modelTen = Lasso(alpha=1.0)
    # define model evaluation method
    cvTen = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(modelTen, Xten, yten, scoring='r2', cv=cvTen, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    print('Mean Absolute Error: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    # tune alpha for 10
    from sklearn.linear_model import LassoCV
    modelTen = LassoCV(alphas=np.arange(0, 1, 0.01), cv=cvTen, n_jobs=-1)
    modelTen.fit(Xten, yten)
    print('alpha: %f' % modelTen.alpha_)

    # predict using tuned alpha
    modelTen = Lasso(alpha=0.03)
    # fit model
    modelTen.fit(Xten, yten)
    # new fake data
    row = [promise5, promise10, 10, 1, 1]
    # make a prediction
    print(modelTen.predict([row])[0])
    predictTen = modelTen.predict([row])
    # summarize prediction
    print('Predicted: %.3f' % predictTen)
    print(modelTen.coef_)
    print(modelTen.intercept_)

    print(type(predictFive))
    predictFive = float(predictFive) + 5
    predictTen = float(predictTen)
    outcomeMap = {10:0, predictFive:5, predictTen:10}
    bestDecision = max(10, predictFive, predictTen)
    print(bestDecision)
    print(outcomeMap[bestDecision])
    return float(outcomeMap[bestDecision])