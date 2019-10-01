import csv
import datetime
import pandas as pd


def check_dataset_validity(fname):
    df = pd.read_csv(fname, header=0)
    hasGpp = 0
    for index, row in df.iterrows():
        site_path = "/Users/kaiweiluo/PycharmProjects/Machine-Learning-NASA/Site_Data_RAW_Collection/" \
                    + row['Site_Name']
        df2 = pd.read_csv(site_path, header=2)
        print("==================")
        print(row['Site_Name'])
        if 'GPP_PI_F' in df2.columns:
            print("Has GPP: True")
            hasGpp += 1
        else:
            print("Has GPP: False")
        print("==================\n")

    print("Number of total sites: " + str(len(df.index)))
    print("Number of sites has GPP: " + str(hasGpp))


def clean_site_date(fname):
    df = pd.read_csv(fname, header=2)
    df = df[['TIMESTAMP_START', 'TIMESTAMP_END', 'TA', 'PPFD_IN']]
    for index, row in df.iterrows():
        if row['TA'] == -9999.0 or row['PPFD_IN'] == -9999.0:
            df.drop(index, inplace=True)
            print(index)
    new_fname = "Clean_" + fname
    df.to_csv(new_fname, index=False)


# clean_site_date("AMF_US-A32_BASE_HH_1-5.csv")
check_dataset_validity("Site_Name.csv")
