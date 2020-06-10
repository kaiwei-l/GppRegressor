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

def pre_processing(path):
    sitelistdf = pd.read_csv("Site-List-2.csv")
    sitelistdf.columns = sitelistdf.columns.str.strip()
    sitelistdf['SiteId'] = sitelistdf['SiteId'].str.strip()

    veg_list = sitelistdf['Veg'].tolist()
    veg_code = list()
    veg_lookup = dict()
    counter = 0
    for veg_type in veg_list:
        if veg_type not in veg_lookup:
            veg_lookup[veg_type] = str(counter)
            counter += 1

    for veg_type in veg_list:
        veg_code.append(veg_lookup[veg_type])

    sitelistdf['veg'] = veg_code
    fname = "Site-List.csv"
    fullname = os.path.join(path, fname)
    sitelistdf.to_csv(fullname, index=False)

def add_features(fname, sitelistdf, outpath):
    siteid = fname.split('/')[-1].split('.')[0]
    siteinfo = sitelistdf.loc[sitelistdf['SiteId'] == siteid]
    veg_type = siteinfo.veg.item()
    elev = siteinfo.Elev.item()
    mat = siteinfo.MAT.item()
    map = siteinfo.MAP.item()
    df = pd.read_csv(fname)
    df['VEG'] = veg_type
    df['ELEV'] = elev
    df['MAT'] = mat
    df['MAP'] = map
    fname = siteid + ".csv"
    fullname = os.path.join(outpath, fname)
    df.to_csv(fullname, index=False)

def site_main(inpath, outpath):
    # Obtain a list of information of sites such as longitude and latitude
    sitelistdf = pd.read_csv("Site-List.csv")
    sitelistdf.columns = sitelistdf.columns.str.strip()
    sitelistdf['SiteId'] = sitelistdf['SiteId'].str.strip()

    directory = os.fsencode(inpath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            file_path = inpath + filename
            p = Process(target=add_features(file_path, sitelistdf, outpath))
            p.start()

# ####################################
# Part 2: Build Machine Learning Model
# ####################################

def merge_df(datapath, outputpath):
    directory = os.fsencode(datapath)
    df_list = list()
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            filepath = datapath + "/" + filename
            df = pd.read_csv(filepath)
            df_list.append(df)
    df = pd.concat(df_list, sort=False)
    df = df.dropna(axis='columns')
    fname = "merged_site.csv"
    fullname = os.path.join(outputpath, fname)
    df.to_csv(fullname, index=False)


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
    figtitle = "Merged_Data"

    df = pd.read_csv(fname)
    feature_cols = ['NIRV']
    # Features selection
    if 'TA_F_MDS' in df.columns:
        feature_cols.append('TA_F_MDS')
    if 'PPFD_IN' in df.columns:
        feature_cols.append('PPFD_IN')
    if 'VPD_F' in df.columns:
        feature_cols.append('VPD_F')
    if 'VEG' in df.columns:
        feature_cols.append('VEG')
    if 'ELEV' in df.columns:
        feature_cols.append('ELEV')
    if 'MAT' in df.columns:
        feature_cols.append('MAT')
    if 'MAP' in df.columns:
        feature_cols.append('MAP')
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
    importances = reg.feature_importances_

    # y_pos = np.arange(len(feature_cols))
    # plt.bar(y_pos, importances, align='center', alpha=0.5)
    # plt.xticks(y_pos, feature_cols)
    # plt.title("Importance of features")
    # output_path = outputdir + figtitle + ".png"
    # plt.savefig(output_path)
    # plt.clf()



inpath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/Processed_Data_Hourly/"
outpath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/new_data/"
datapath = "/Users/kaiweiluo/PycharmProjects/GppRegressor/new_data/"
fname = "/Users/kaiweiluo/PycharmProjects/GppRegressor/merged_site.csv"
fig_output = "/Users/kaiweiluo/PycharmProjects/GppRegressor/results/"
# random_forest_estimate(fname, fig_output)
# pre_processing("/Users/kaiweiluo/PycharmProjects/GppRegressor/")
# site_main(inpath, outpath)
# merge_df(datapath, outpath)
random_forest_estimate(fname, fig_output)

