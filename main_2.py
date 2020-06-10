import csv
import datetime
import re
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats
from multiprocessing import Process
import sys

# ####################################
# Author: Kaiwei Luo
# Part 1: Data processing and cleaning
# ####################################
def row_aux(row):
    val = str(row['TIMESTAMP_END'])[0:8]
    return val

def site_sanitizer_aux(fname, outputpath):
    site_id = fname.split('/')[-1].split('.')[0]
    print(site_id)
    df = pd.read_csv(fname)
    df['TIMESTAMP'] = df.apply(row_aux, axis=1)
    df = df.groupby(['TIMESTAMP']).mean().reset_index()
    df = df.drop(['TIMESTAMP_START', 'TIMESTAMP_END'], axis=1)
    f = site_id + ".csv"
    df.to_csv(os.path.join(outputpath, f), index=False)


def site_sanitizer(inpath, outpath):
    # Obtain a list of information of sites such as longitude and latitude
    directory = os.fsencode(inpath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            file_path = inpath + filename
            site_sanitizer_aux(file_path, outpath)

# ####################################
# Part 2: Build Machine Learning Model
# ####################################


def draw_graph(reg, X_test, y_test, atitle, outputdir):
    y_predict = reg.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_predict)
    r2_score_test = r2_score(y_test, y_predict)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_predict)
    line = slope * y_test + intercept
    plt.plot(y_test, y_predict, 'o', y_test, line)
    title = atitle + " rmse: " + str(rmse_test) + " r2 score: " + str(r2_score_test)
    plt.xlabel("test data")
    plt.ylabel("predict data")
    plt.title(title)
    output_path = outputdir + atitle + ".png"
    plt.savefig(output_path)
    #plt.show()
    plt.clf()


def random_forest_estimate(fname, outputdir):
    # Loading data
    figtitle = os.path.basename(fname).split('.')[0]

    df = pd.read_csv(fname)
    feature_cols = ['NIRV']
    # Features selection
    # if 'TA_F_MDS' in df.columns:
    #     feature_cols.append('TA_F_MDS')
    # if 'PPFD_IN' in df.columns:
    #     feature_cols.append('PPFD_IN')
    if 'VPD_F' in df.columns:
        feature_cols.append('VPD_F')
    X = df.loc[:, feature_cols]
    y = df['GPP_NT_VUT_MEAN']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training classifiers
    # 1. Random Forest
    gridsearch_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'max_depth': range(3, 7),
                                                                                    'n_estimators': (
                                                                                        10, 50, 100, 1000)},
                                     cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_result_forest = gridsearch_forest.fit(X_train, y_train)
    best_params_forest = grid_result_forest.best_params_
    reg = RandomForestRegressor(max_depth=best_params_forest["max_depth"],
                                n_estimators=best_params_forest["n_estimators"],
                                random_state=False, verbose=False)
    scores = cross_val_score(reg, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
    print("scores: " + str(scores))
    reg.fit(X_train, y_train)
    draw_graph(reg, X_test, y_test, figtitle, outputdir)


def regression_controller(datapath, outputpath):
    directory = os.fsencode(datapath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            filepath = datapath + "/" + filename
            random_forest_estimate(filepath, outputpath)


inpath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/Processed_Data_Hourly/"
outpath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/new_data/"
dataset_dir = "/Users/kaiweiluo/PycharmProjects/GppRegressor/new_data/"
fig_output = "/Users/kaiweiluo/PycharmProjects/GppRegressor/results/"
#site_sanitizer(inpath, outpath)
regression_controller(dataset_dir, fig_output)

