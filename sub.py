import csv
import datetime
import re
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats

# ####################################
# Author: Kaiwei Luo
# Part 1: Data processing and cleaning
# ####################################


def check_dataset_validity():
    directory = os.fsencode("/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/Site-CSV/WET")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            file_path = "/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/Site-CSV/WET/" + filename
            df = pd.read_csv(file_path, header=2)
            if 'GPP_PI_F' in df.columns and 'PPFD_IN' in df.columns:
                print("==================")
                print(filename)
                print("Has GPP and PPFD_IN: True")
                print("==================\n")


def timestamp_update(timestamp, hrs):
    date_add_hrs = datetime.datetime(int(timestamp[0:4]), int(timestamp[4:6]), int(timestamp[6:8]),
                                     int(timestamp[8:10]), int(timestamp[10:12])) + \
                   datetime.timedelta(hours=hrs)
    new_date = str(date_add_hrs.year) + str(date_add_hrs.month).zfill(2) + str(date_add_hrs.day).zfill(2) + \
               str(date_add_hrs.hour).zfill(2) + \
               str(date_add_hrs.minute).zfill(2)
    return new_date


def site_sanitizer_aux(fname, epic_prefix, sitelistdf, outputpath):
    # ===========================
    # Issue 1: PPFD not found: Solved
    # Issue 2: GPP_PI_F_1_2, Multiple name, OWN
    # ===========================

    with open(fname, newline='') as f:
        reader = csv.reader(f)
        siteid = next(reader)[0].split()[2]

    siteinfo = sitelistdf.loc[sitelistdf['Site Id'] == siteid]
    lon = float(siteinfo['Long'])
    hrs = -math.ceil(((-7.5 - lon) / 15))
    epic_file = epic_prefix + siteid + ".csv"

    df = pd.read_csv(fname, header=2)
    epicdf = pd.read_csv(epic_file, header=None)
    epicdf.columns = ["Year", "Month", "Day", "Hour", "Minute", "Second", "Blue", "Red", "NIRS"]
    hasvpd = False
    hasppfd = False
    hasgpp = False
    rgpp = re.compile("[gG][pP][pP]")
    rppfd = re.compile("^[pP][pP][fF][dD]_[iI][nN]$")
    rvpd = re.compile("^[vV][pP][dD]_[pP][iI]$")
    cols = list(df.columns)
    for col in cols:
        if re.search(rgpp, col):
            hasgpp = True
        if re.search(rppfd, col):
            hasppfd = True
        if re.search(rvpd, col):
            hasvpd = True

    # Sanitizing Epic dataset
    nirv_list = []
    epic_date_list = []
    for index, row in epicdf.iterrows():
        print("EPIC " + siteid + " " + str(index))
        if row["Year"] == 0 or row["Month"] == 0 or row["Day"] == 0 or row["Hour"] == 0 or row["Minute"] == 0 or row[
                "Second"] == 0 or row["Blue"] == 0 or row["Red"] == 0 or row["NIRS"] == 0:
            epicdf.drop(index, inplace=True)
        else:
            nirv = (row['NIRS'] - row['Red']) / (row['NIRS'] + row['Red']) * row['NIRS']
            nirv_list.append(nirv)
            dateobj = datetime.datetime(int(row['Year']), int(row['Month']), int(row['Day']),
                                        int(row['Hour']), int(row['Minute']))
            new_date = str(dateobj.year) + str(dateobj.month).zfill(2) + str(dateobj.day).zfill(2) + \
                       str(dateobj.hour).zfill(2) + str(dateobj.minute).zfill(2)
            epic_date_list.append(new_date)
    epicdf['NIRV'] = nirv_list
    epicdf['Datetime'] = epic_date_list

    if hasgpp and hasppfd:
        if not hasvpd:
            df["VPD_PI"] = ""
        df = df[['TIMESTAMP_START', 'TIMESTAMP_END', 'TA', 'PPFD_IN', 'VPD_PI', 'GPP_PI_F', 'RH']]
        for index, row in df.iterrows():
            print("Site " + siteid + " " + str(index))
            if int(row['TIMESTAMP_START']) >= 201501010000:  # Check timestamp validity, must be later than 2015
                if row['TA'] == -9999.0 or row['PPFD_IN'] == -9999.0 or row['GPP_PI_F'] == -9999.0 \
                        or row['VPD_PI'] == -9999.0:
                    df.drop(index, inplace=True)
                else:
                    # Adjust timestamp to UTC
                    new_start_datetime = timestamp_update(str(row['TIMESTAMP_START']), hrs)
                    new_end_datetime = timestamp_update(str(row['TIMESTAMP_END']), hrs)
                    df.at[index, 'TIMESTAMP_START'] = new_start_datetime
                    df.at[index, 'TIMESTAMP_END'] = new_end_datetime

                    # Calculate VPD_PI
                    if not hasvpd:
                        svp = 610.7 * 10 ** ((7.5 * row['TA']) / (237.3 + row['TA'])) * 0.01
                        vpd = ((100 - row['RH']) / 100) * svp
                        df.at[index, 'VPD_PI'] = vpd
            else:
                df.drop(index, inplace=True)
        df.drop('RH', axis=1, inplace=True)

        # Merge dataframe
        data = []
        site_iter = df.iterrows()
        epic_iter = epicdf.iterrows()
        site_row = next(site_iter)
        epic_row = next(epic_iter)
        try:
            while site_row is not None and epic_row is not None:
                if int(site_row[1]['TIMESTAMP_START']) <= int(epic_row[1]['Datetime']) <= int(site_row[1]['TIMESTAMP_END']):
                    row = site_row[1].tolist()
                    row.append(epic_row[1]['NIRV'])
                    data.append(row)
                    site_row = next(site_iter)
                    epic_row = next(epic_iter)
                else:
                    if int(epic_row[1]['Datetime']) < int(site_row[1]['TIMESTAMP_START']):
                        epic_row = next(epic_iter)
                    site_row = next(site_iter)
        except Exception as e:
            print(e)
        mergedf = pd.DataFrame(data, columns=['TIMESTAMP_START', 'TIMESTAMP_END', 'TA', 'PPFD_IN', 'VPD_PI',
                                              'GPP_PI_F', 'NIRV'])
        new_fname = "Clean_" + siteid + ".csv"
        mergedf.to_csv(os.path.join(outputpath, new_fname), index=False)


def site_sanitizer(inpath, outpath, epicpath):
    # Obtain a list of information of sites such as longitude and latitude
    sitelistdf = pd.read_csv("Site-List.csv")
    sitelistdf.columns = sitelistdf.columns.str.strip()
    sitelistdf['Site Id'] = sitelistdf['Site Id'].str.strip()

    directory = os.fsencode(inpath)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            file_path = inpath + filename
            site_sanitizer_aux(file_path, epicpath, sitelistdf, outpath)

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
    plt.text(17, 5, "rmse: " + str(rmse_test))
    plt.text(17, 3, "r2 score: " + str(r2_score_test))
    plt.xlabel("test data")
    plt.ylabel("predict data")
    plt.title(atitle)
    output_path = outputdir + atitle + ".png"
    plt.savefig(output_path)
    plt.show()


def random_forest_estimate(fname, outputdir):
    # Loading data
    figtitle = os.path.basename(fname)
    figtitle = os.path.splitext(figtitle)[0][6:]
    df = pd.read_csv(fname)
    feature_cols = ['TA', 'VPD_PI', 'PPFD_IN', 'NIRV']
    X = df.loc[:, feature_cols]
    y = df['GPP_PI_F']
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


inpath = "/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/Site-CSV/WET/"
outpath = "/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/DATASET-WET"
epicpath = "/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/EPIC_data_update/"
dataset_dir = "/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/DATASET-WET"
fig_output = "/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/DATASET-WET/"
# site_sanitizer(inpath, outpath, epicpath)
regression_controller(dataset_dir, fig_output)
