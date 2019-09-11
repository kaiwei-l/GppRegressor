import csv
import datetime
import pandas as pd


# ####################################
# Part 1: Data processing and cleaning
# ####################################

# Cleaning site raw data
def clean_site():
    with open("AMF_US-Var_BASE_HH_13-5.csv", 'r') as f:
        source = csv.reader(f)
        with open("Clean_Site_Data.csv", 'w') as result:
            wtr = csv.writer(result)
            for r in source:
                try:
                    if int(r[0]) >= 201501010000:
                        if float(r[17]) != -9999 and float(r[18]) != -9999 and float(r[34]) != -9999 and \
                                float(r[43]) != -9999:
                            start_date_add_eight = datetime.datetime(int(r[0][0:4]), int(r[0][4:6]), int(r[0][6:8]),
                                                                     int(r[0][8:10]), int(r[0][10:12])) + \
                                                   datetime.timedelta(hours=8)
                            new_start_date = str(start_date_add_eight.year) + str(start_date_add_eight.month).zfill(2) \
                                             + str(start_date_add_eight.day).zfill(2) + \
                                             str(start_date_add_eight.hour).zfill(2) + \
                                             str(start_date_add_eight.minute).zfill(2)
                            end_date_add_eight = datetime.datetime(int(r[1][0:4]), int(r[1][4:6]), int(r[1][6:8]),
                                                                   int(r[1][8:10]), int(r[1][10:12])) + \
                                                 datetime.timedelta(hours=8)
                            new_end_date = str(end_date_add_eight.year) + str(end_date_add_eight.month).zfill(2) + \
                                           str(end_date_add_eight.day).zfill(2) + str(
                                end_date_add_eight.hour).zfill(2) + \
                                           str(end_date_add_eight.minute).zfill(2)
                            wtr.writerow((new_start_date, new_end_date, r[17], r[18], r[34], r[43]))
                except ValueError:
                    if len(r) > 1:
                        wtr.writerow((r[0], r[1], r[17], r[18], r[34], r[43]))
                    else:
                        wtr.writerow(r)


# Cleaning EPIC data
def clean_epic():
    with open("US_VAR_EPIC_data.csv", 'r') as f:
        source = csv.reader(f)
        with open("Clean_EPIC_Data.csv", 'w') as result:
            wtr = csv.writer(result)
            for r in source:
                valid_row = True
                for e in r:
                    if e == "NaN":
                        valid_row = False
                        break
                if valid_row:
                    try:
                        ndvi = (float(r[8]) - float(r[7])) / (float(r[8]) + float(r[7]))
                        date_entry = r[0] + r[1].zfill(2) + r[2].zfill(2) + r[3].zfill(2) + r[4].zfill(2)
                        wtr.writerow([date_entry, ndvi])
                    except ValueError:
                        wtr.writerow(["TIMESTAMP", "NDVI"])


def data_set_gen():
    # Combine EPIC and site data
    with open("Clean_EPIC_Data.csv", 'r') as f1:
        epic_reader = csv.reader(f1)
        with open("Clean_Site_Data.csv", 'r') as f2:
            site_reader = csv.reader(f2)
            with open("dataset.csv", 'w') as result:
                wtr = csv.writer(result)
                next(epic_reader)
                wtr.writerow(next(site_reader))
                wtr.writerow(next(site_reader))
                indx_line = next(site_reader)
                indx_line.insert(5, "NDVI")
                wtr.writerow(indx_line)
                for epic_row in epic_reader:
                    for site_row in site_reader:
                        if (float(epic_row[0]) >= float(site_row[0])) and (float(epic_row[0]) <= float(site_row[1])):
                            wtr.writerow((site_row[0], site_row[1], site_row[2], site_row[3], site_row[4],
                                          epic_row[1], site_row[5]))
                            break
                        else:
                            pass


# clean_site()
# data_set_gen()

# ####################################
# Part 2: Build Machine Learning Model
# ####################################
