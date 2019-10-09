import csv
import datetime
import re
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


def site_sanitizer(fname):
    # ===========================
    # Issue 1: PPFD not found
    # ===========================

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
                    if not hasvpd:
                        svp = 610.7 * 10 ** ((7.5 * row['TA']) / (237.3 + row['TA'])) * 0.01
                        vpd = ((100 - row['RH']) / 100) * svp
                        df.at[index, 'VPD_PI'] = vpd
            else:
                df.drop(index, inplace=True)
        df.drop('RH', axis=1)
        new_fname = "Clean_" + fname
        df.to_csv(new_fname, index=False)


# check_dataset_validity()
site_sanitizer("AMF_US-OWC_BASE_HH_2-5.csv")
