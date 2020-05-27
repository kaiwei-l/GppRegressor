import csv
import datetime
import re
import math
import os
import pandas as pd
import numpy as np
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


def evaluate_results(reg, X_test, y_test, atitle):
    y_predict = reg.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_predict)
    r2_score_test = r2_score(y_test, y_predict)
    title = atitle + " rmse: " + str(rmse_test) + " r2 score: " + str(r2_score_test)
    print(title)
    return rmse_test, r2_score_test


def random_forest_estimate(fname, outputdir1, outputdir2):
    # Loading data
    figtitle = os.path.basename(fname).split('.')[0]

    features = ("NIRV", "NIRV, PPFD", "NIRV, VPD", "NIRV, TA, PPFD, VPD")
    y_pos = np.arange(len(features))
    rmse_list = list()
    r2_list = list()

    df = pd.read_csv(fname)
    feature_cols = ['NIRV']
    # Features selection
    # if 'TA_F_MDS' in df.columns:
    #     feature_cols.append('TA_F_MDS')
    # if 'PPFD_IN' in df.columns:
    #     feature_cols.append('PPFD_IN')
    # if 'VPD_F' in df.columns:
    #     feature_cols.append('VPD_F')
    X = df.loc[:, feature_cols]
    y = df['GPP_NT_VUT_MEAN']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training classifiers
    # 1. Features: NIRV
    gridsearch_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid={'max_depth': range(3, 7),
                                                                                    'n_estimators': (
                                                                                        10, 50, 100, 1000)},
                                     cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_result_forest = gridsearch_forest.fit(X_train, y_train)
    best_params_forest = grid_result_forest.best_params_
    reg = RandomForestRegressor(max_depth=best_params_forest["max_depth"],
                                n_estimators=best_params_forest["n_estimators"],
                                random_state=False, verbose=False)
    reg.fit(X_train, y_train)
    (rmse, r2) = evaluate_results(reg, X_test, y_test, figtitle)
    rmse_list.append(rmse)
    r2_list.append(r2)

    # 2. Features: NIRV, PPFD
    feature_cols = ['NIRV']
    if 'PPFD_IN' in df.columns:
        feature_cols.append('PPFD_IN')
    X = df.loc[:, feature_cols]
    y = df['GPP_NT_VUT_MEAN']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    gridsearch_forest = GridSearchCV(estimator=RandomForestRegressor(),
                                     param_grid={'max_depth': range(3, 7),
                                                 'n_estimators': (
                                                     10, 50, 100, 1000)},
                                     cv=5, scoring='neg_mean_squared_error',
                                     verbose=0, n_jobs=-1)
    grid_result_forest = gridsearch_forest.fit(X_train, y_train)
    best_params_forest = grid_result_forest.best_params_
    reg = RandomForestRegressor(max_depth=best_params_forest["max_depth"],
                                n_estimators=best_params_forest["n_estimators"],
                                random_state=False, verbose=False)
    reg.fit(X_train, y_train)
    (rmse, r2) = evaluate_results(reg, X_test, y_test, figtitle)
    rmse_list.append(rmse)
    r2_list.append(r2)

    # 3. Features: NIRV, VPD
    feature_cols = ['NIRV']
    if 'VPD_F' in df.columns:
        feature_cols.append('VPD_F')
    X = df.loc[:, feature_cols]
    y = df['GPP_NT_VUT_MEAN']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    gridsearch_forest = GridSearchCV(estimator=RandomForestRegressor(),
                                     param_grid={'max_depth': range(3, 7),
                                                 'n_estimators': (
                                                     10, 50, 100, 1000)},
                                     cv=5, scoring='neg_mean_squared_error',
                                     verbose=0, n_jobs=-1)
    grid_result_forest = gridsearch_forest.fit(X_train, y_train)
    best_params_forest = grid_result_forest.best_params_
    reg = RandomForestRegressor(max_depth=best_params_forest["max_depth"],
                                n_estimators=best_params_forest["n_estimators"],
                                random_state=False, verbose=False)
    reg.fit(X_train, y_train)
    (rmse, r2) = evaluate_results(reg, X_test, y_test, figtitle)
    rmse_list.append(rmse)
    r2_list.append(r2)

    # 4. Features: NIRV, TA, PPFD, VPD
    feature_cols = ['NIRV']
    if 'TA_F_MDS' in df.columns:
        feature_cols.append('TA_F_MDS')
    if 'PPFD_IN' in df.columns:
        feature_cols.append('PPFD_IN')
    if 'VPD_F' in df.columns:
        feature_cols.append('VPD_F')

    X = df.loc[:, feature_cols]
    y = df['GPP_NT_VUT_MEAN']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    gridsearch_forest = GridSearchCV(estimator=RandomForestRegressor(),
                                     param_grid={'max_depth': range(3, 7),
                                                 'n_estimators': (
                                                     10, 50, 100, 1000)},
                                     cv=5, scoring='neg_mean_squared_error',
                                     verbose=0, n_jobs=-1)
    grid_result_forest = gridsearch_forest.fit(X_train, y_train)
    best_params_forest = grid_result_forest.best_params_
    reg = RandomForestRegressor(max_depth=best_params_forest["max_depth"],
                                n_estimators=best_params_forest["n_estimators"],
                                random_state=False, verbose=False)
    reg.fit(X_train, y_train)
    (rmse, r2) = evaluate_results(reg, X_test, y_test, figtitle)
    rmse_list.append(rmse)
    r2_list.append(r2)

    plt.bar(y_pos, rmse_list, align='center', alpha=0.5)
    plt.xticks(y_pos, features)
    plt.ylabel('RMSE')
    plt.title(figtitle)
    output_path = outputdir1 + figtitle + ".png"
    plt.savefig(output_path)
    plt.clf()

    plt.bar(y_pos, r2_list, align='center', alpha=0.5)
    plt.xticks(y_pos, features)
    plt.ylabel('r2')
    plt.title(figtitle)
    output_path = outputdir2 + figtitle + ".png"
    plt.savefig(output_path)
    plt.clf()


def regression_controller(datapath, outputpath1, outputpath2):
    directory = os.fsencode(datapath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            filepath = datapath + "/" + filename
            random_forest_estimate(filepath, outputpath1, outputpath2)



dataset_dir = "/Users/kaiweiluo/PycharmProjects/GppRegressor/Processed_Data_Daily"
fig_output1 = "/Users/kaiweiluo/PycharmProjects/GppRegressor/Bar_Chart_Daily/RMSE/"
fig_output2 = "/Users/kaiweiluo/PycharmProjects/GppRegressor/Bar_Chart_Daily/R2/"
regression_controller(dataset_dir, fig_output1, fig_output2)
