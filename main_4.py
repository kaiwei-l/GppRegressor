# MODIS

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
import numpy as np
from multiprocessing import Process
import sys

# ####################################
# Author: Kaiwei Luo
# Part 1: Data processing and cleaning
# ####################################


def timestamp_update(timestamp, hrs):
    date_add_hrs = datetime.datetime(int(timestamp[0:4]), int(timestamp[4:6]), int(timestamp[6:8]),
                                     int(timestamp[8:10]), int(timestamp[10:12])) + \
                   datetime.timedelta(hours=hrs)
    new_date = str(date_add_hrs.year) + str(date_add_hrs.month).zfill(2) + str(date_add_hrs.day).zfill(2) + \
               str(date_add_hrs.hour).zfill(2) + \
               str(date_add_hrs.minute).zfill(2)
    return new_date


def site_sanitizer_aux(siteDataPath, modisPath, sitelistdf, outputpath, scale, algoName):
    siteid = siteDataPath.split('/')[-1].split('_')[1]

    siteinfo = sitelistdf.loc[sitelistdf['SiteId'] == siteid]
    lon = float(siteinfo['Longitude'])
    hrs = -math.ceil(((-7.5 - lon) / 15))
    modisPath = modisPath + scale + "/" + siteid + "/" + scale + "_" + algoName + "_" + siteid + ".csv"

    df = pd.read_csv(siteDataPath)
    modisdf = pd.read_csv(modisPath)

    # Sanitizing Epic dataset
    nirv_list = []
    modis_date_list = []
    print("I. MODIS " + siteid)

    modisdf.dropna(inplace=True)  # Clean MODIS data

    for index, row in modisdf.iterrows():
        nirv = (row['sur_refl_b02'] - row['sur_refl_b01']) / (row['sur_refl_b02'] + row['sur_refl_b01']) * row['sur_refl_b02']
        nirv_list.append(nirv)
        date = row['date'].replace("-", "")
        modis_date_list .append(date)
    modisdf['NIRV'] = nirv_list
    modisdf['Datetime'] = modis_date_list

    # Sanitizing site data
    dfIndex = ['TIMESTAMP_START', 'TIMESTAMP_END']
    if 'TA_F_MDS' in df.columns:
        dfIndex.append('TA_F_MDS')
    if 'PPFD_IN' in df.columns:
        dfIndex.append('PPFD_IN')
    if 'VPD_F' in df.columns:
        dfIndex.append('VPD_F')
    if 'GPP_NT_VUT_MEAN' in df.columns:
        dfIndex.append('GPP_NT_VUT_MEAN')
    if 'GPP_NT_CUT_MEAN' in df.columns:
        dfIndex.append('GPP_NT_CUT_MEAN')
    if 'GPP_DT_VUT_MEAN' in df.columns:
        dfIndex.append('GPP_DT_VUT_MEAN')
    if 'GPP_DT_CUT_MEAN' in df.columns:
        dfIndex.append('GPP_DT_CUT_MEAN')

    # dataset has no useable data
    if len(dfIndex) <= 2:
        return

    df = df[dfIndex]
    df = df[df['TIMESTAMP_START'] >= 201501010000]
    for i in range(2, len(dfIndex)):
        df = df[df[dfIndex[i]] != -9999.0]
    print("II. Site " + siteid)
    for index, row in df.iterrows():
        # Adjust timestamp to UTC
        new_start_datetime = timestamp_update(str(row['TIMESTAMP_START']), hrs)
        new_end_datetime = timestamp_update(str(row['TIMESTAMP_END']), hrs)
        df.at[index, 'TIMESTAMP_START'] = new_start_datetime
        df.at[index, 'TIMESTAMP_END'] = new_end_datetime

    # Merge dataframe
    data = []
    site_iter = df.iterrows()
    modis_iter = modisdf.iterrows()
    site_row = next(site_iter)
    modis_row = next(modis_iter)
    print("III. Merging...")

    try:
        while site_row is not None and modis_row is not None:
            if int(str(int(site_row[1]['TIMESTAMP_START']))[:8]) <= int(modis_row[1]['Datetime']) <= int(str(int(site_row[1]['TIMESTAMP_END']))[:8]):
                row = site_row[1].tolist()
                row.append(modis_row[1]['NIRV'])
                data.append(row)
                site_row = next(site_iter)
                modis_row = next(modis_iter)
            else:
                if int(modis_row[1]['Datetime']) < int(str(int(site_row[1]['TIMESTAMP_START']))[:8]):
                    modis_row = next(modis_iter)
                site_row = next(site_iter)
    except Exception as e:
        print(e)
    dfIndex.append('NIRV') # columns list for merged df
    mergedf = pd.DataFrame(data, columns=dfIndex)
    new_fname = scale + "_" + algoName + "_" + siteid + ".csv"
    mergedf.to_csv(os.path.join(outputpath, new_fname), index=False)


# scale: 3km or 5km
def site_sanitizer(inputPath, outputPath, modisPath, scale, algoName):
    # Obtain a list of information of sites such as longitude and latitude
    sitelistdf = pd.read_csv("Site-List.csv")
    sitelistdf.columns = sitelistdf.columns.str.strip()
    sitelistdf['SiteId'] = sitelistdf['SiteId'].str.strip()

    directory = os.fsencode(inputPath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            siteDataPath = inputPath + filename
            site_sanitizer_aux(siteDataPath, modisPath, sitelistdf, outputPath, scale, algoName)


# ####################################
# Part 2: Build Machine Learning Model
# ####################################


def draw_graph(reg, X_test, y_test, siteName, scale, algoName, outputdir):
    y_predict = reg.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_predict)
    r2_score_test = r2_score(y_test, y_predict)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_predict)
    line = slope * y_test + intercept
    plt.plot(y_test, y_predict, 'o', y_test, line)
    plt.rcParams["axes.titlesize"] = 8
    title = scale + " " + algoName + " " + siteName + " rmse: " + str(rmse_test) + " r2 score: " + str(r2_score_test)
    plt.xlabel("test data")
    plt.ylabel("predict data")
    plt.title(title)
    output_path = outputdir + scale + "_" + algoName + "_" + siteName + ".png"
    plt.savefig(output_path)
    #plt.show()
    plt.clf()


def random_forest_estimate(fname, outputdir, scale, algoName, siteId):
    # Loading data

    df = pd.read_csv(fname)
    feature_cols = ['NIRV']
    # Features selection
    if 'TA_F_MDS' in df.columns:
        feature_cols.append('TA_F_MDS')
    if 'PPFD_IN' in df.columns:
        feature_cols.append('PPFD_IN')
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
    draw_graph(reg, X_test, y_test, siteId, scale, algoName, outputdir)


def regression_controller(datapath, outputpath):
    directory = os.fsencode(datapath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            filename_list = filename.split('_')
            scale = filename_list[0]
            algoName = filename_list[1]
            siteId = filename_list[2].split('.')[0]
            filepath = datapath + "/" + filename
            random_forest_estimate(filepath, outputpath, scale, algoName, siteId)


inputPath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/data/"
outputPath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/new_data/"
modisPath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/MODIS/"
# site_sanitizer(inputPath, outputPath, modisPath, "3km", "MYD")
datasetPath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/new_data"
figOutputPath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/results/"
regression_controller(datasetPath, figOutputPath)