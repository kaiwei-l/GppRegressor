import csv
import datetime
import re
import math
import os
import pandas as pd


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


def site_sanitizer_aux(fname, sitelistdf):
    # ===========================
    # Issue 1: PPFD not found
    # ===========================
    # sitelistdf = pd.read_csv("Site-List.csv")
    # sitelistdf.columns = sitelistdf.columns.str.strip()
    # sitelistdf['Site Id'] = sitelistdf['Site Id'].str.strip()
    # sitelistdf_obj = sitelistdf.select_dtypes(['object'])
    # sitelistdf[sitelistdf_obj.columns] = sitelistdf_obj.apply(lambda x: x.str.strip())

    with open(fname, newline='') as f:
        reader = csv.reader(f)
        siteid = next(reader)[0].split()[2]

    siteinfo = sitelistdf.loc[sitelistdf['Site Id'] == siteid]
    lon = float(siteinfo['Long'])
    hrs = -math.ceil(((-7.5 - lon) / 15))

    df = pd.read_csv(fname, header=2)
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
    if hasgpp and hasppfd:
        if not hasvpd:
            df["VPD_PI"] = ""
        df = df[['TIMESTAMP_START', 'TIMESTAMP_END', 'TA', 'PPFD_IN', 'VPD_PI', 'GPP_PI_F', 'RH']]
        for index, row in df.iterrows():
            if int(row['TIMESTAMP_START']) >= 201501010000:  # Check timestamp validity, must be later than 2015
                if row['TA'] == -9999.0 or row['PPFD_IN'] == -9999.0 or row['GPP_PI_F'] == -9999.0 \
                        or row['VPD_PI'] == -9999.0:
                    df.drop(index, inplace=True)
                else:
                    # Calculate VPD_PI
                    if not hasvpd:
                        svp = 610.7 * 10 ** ((7.5 * row['TA']) / (237.3 + row['TA'])) * 0.01
                        vpd = ((100 - row['RH']) / 100) * svp
                        df.at[index, 'VPD_PI'] = vpd
                    # Adjust timestamp to UTC
                    new_start_datetime = timestamp_update(str(row['TIMESTAMP_START']), hrs)
                    new_end_datetime = timestamp_update(str(row['TIMESTAMP_END']), hrs)
                    df.at[index, 'TIMESTAMP_START'] = new_start_datetime
                    df.at[index, 'TIMESTAMP_END'] = new_end_datetime
            else:
                df.drop(index, inplace=True)
        df.drop('RH', axis=1, inplace=True)
        new_fname = "Clean_" + fname
        df.to_csv(new_fname, index=False)


def site_sanitizer():
    # Obtain a list of information of sites such as longitude and latitude
    sitelistdf = pd.read_csv("Site-List.csv")
    sitelistdf.columns = sitelistdf.columns.str.strip()
    sitelistdf['Site Id'] = sitelistdf['Site Id'].str.strip()

    directory = os.fsencode("/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/Site-CSV/WET")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            file_path = "/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/Site-CSV/WET/" + filename
            print(filename)
            site_sanitizer_aux(file_path, sitelistdf)


# check_dataset_validity()
# site_sanitizer_aux("AMF_US-OWC_BASE_HH_2-5.csv")
site_sanitizer()
