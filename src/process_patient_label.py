"""
Author: -
Email: -
Last Modified: Apr, 2021

Generate dataframes with columns ["user_id", "timestamp", "label"] for each classification task.

"""

import numpy as np
import pandas as pd
import mysql.connector
from tqdm import tqdm
from datetime import timedelta
from utils.query_from_db import *
from utils.pandas_operations import *
import random

def get_df_visits(start_date_str, train_end_date_str):
    query1 = 'SELECT vid, pid, adate, ddate, mortality, severity from visits where ddate >= "{}" and adate < "{}"'.format(start_date_str, train_end_date_str)
    column_names, result = query_from_db(query1)
    df_visits = pd.DataFrame(result, columns=column_names)

    df_visits = df_visits.sort_values(by="adate")
    df_visits.drop_duplicates(subset="pid", keep="last", inplace=True)

    return df_visits

def gen_df_mortality_severity(df_visits):
    # Some instances do not have value for severity and mortality
    df_visits_temp = df_visits.dropna()

    mapping_dict = {"Minor": 0, "Moderate":1, "Major":2, "Extreme":3}

    df_mortality_severity = df_visits_temp[["pid", "mortality", "severity"]]
    df_mortality_severity = df_mortality_severity.rename(columns={"pid": "user_id"})

    df_mortality_severity.insert(df_mortality_severity.shape[1], column="label_mortality", value=df_mortality_severity["mortality"].map(mapping_dict))
    df_mortality_severity.insert(df_mortality_severity.shape[1], column="label_severity", value=df_mortality_severity["severity"].map(mapping_dict))

    df_mortality = df_mortality_severity[["user_id", "label_mortality"]]
    df_severity = df_mortality_severity[["user_id", "label_severity"]]
    return df_mortality.rename(columns={"label_mortality": "label"}), df_severity.rename(columns={"label_severity": "label"})

def gen_df_CDI(df_visits):
    query_cdiff = 'SELECT vid, cdate from cdiff'
    column_names, result = query_from_db(query_cdiff)
    df_cdiff = pd.DataFrame(result, columns=column_names)

    df_cdiff = pd.merge(left=df_cdiff, right=df_visits[["vid", "pid"]], how="inner", on="vid")[["pid", "cdate"]]
    # Predict one CDI case only per patient
    df_cdiff.sort_values(by="cdate", inplace=True)
    df_cdiff.drop_duplicates(subset="pid", keep="last", inplace=True)
    # Get the date of 3 days before CDI test date
    df_cdiff.insert(loc=1, column="date", value = df_cdiff.cdate - pd.Timedelta(days=3))
    df_cdiff.insert(loc=1, column="date_timestamp", value = pd.to_datetime(df_cdiff.date.dt.date))
    # df_cdiff = df_cdiff.astype({"date_timestamp": })
    df_cdiff = df_cdiff.rename(columns={"pid": "user_id"})

    # Filter in only those dates in the start - end timeframe.
    df_cdiff = filter_records(df_cdiff, start_date, train_end_date)

    df_CDI = df_cdiff[["user_id", "date_timestamp"]]
    df_CDI.insert(loc=df_CDI.shape[1], value=1, column="label")
    CDI_user_array = df_CDI.user_id.unique()

    df_visits_nonCDI = df_visits[~df_visits["pid"].isin(CDI_user_array)]

    ###################
    # For non-CDI patients, get a random date between start_date and train_end_date that is wihtin their hospitalization period
    df_visits_nonCDI.insert(column="start", loc=4, value=start_date)
    df_visits_nonCDI.insert(column="end", loc=5, value=train_end_date -  pd.Timedelta(days=1))

    df_visits_nonCDI.insert(column="start_rand", loc=1, value=pd.to_datetime(df_visits_nonCDI[["adate", "start"]].max(axis=1).dt.date))
    df_visits_nonCDI.insert(column="end_rand", loc=2, value=pd.to_datetime(df_visits_nonCDI[["ddate", "end"]].min(axis=1).dt.date))
    df_visits_nonCDI.insert(column="days", loc=3, value=(df_visits_nonCDI.end_rand - df_visits_nonCDI.start_rand).dt.days)

    days_to_add = [random.randint(0, days) for days in df_visits_nonCDI.days.values]
    df_visits_nonCDI.insert(column="days_to_add", value=days_to_add, loc=3)

    df_visits_nonCDI.insert(column="date_timestamp", loc=1, value=pd.to_datetime((df_visits_nonCDI["start_rand"] + pd.to_timedelta(df_visits_nonCDI["days_to_add"], unit='d')).dt.date))
    ###################
    # get df_nonCDI
    df_visits_nonCDI = df_visits_nonCDI.rename(columns={"pid": "user_id"})
    df_nonCDI = df_visits_nonCDI[["user_id", "date_timestamp"]]
    df_nonCDI.insert(loc=df_nonCDI.shape[1], value=0, column="label")

    # Add df_nonCDI to df_CDI
    df_CDI = pd.concat([df_CDI, df_nonCDI])
    df_CDI.insert(column="start_date", value=start_date, loc=df_CDI.shape[1])
    # print(df_CDI.dtypes)
    df_CDI.insert(column="timestamp", loc=1, value=(df_CDI["date_timestamp"] - df_CDI["start_date"]).dt.days)

    # df_CDI = df_CDI.assign(timestamp=lambda x: (x.date_timestamp - start_date).dt.days)
    df_CDI = df_CDI[["user_id", "timestamp", "label"]]

    return df_CDI

def gen_df_MICU_transfer(df_visits):
    uid_MICU = 12

    # Query transfer records from non-MICU to MICU
    query1 = 'SELECT * from transfers where srcuid!={} and dstuid={} and tdate >= "{}" and tdate < "{}"'.format(uid_MICU, uid_MICU, start_date_str, train_end_date_str)
    column_names, result = query_from_db(query1)
    df_transfers = pd.DataFrame(result, columns=column_names)

    # Add a column that has day since start date
    df_transfers = df_transfers.assign(timestamp=lambda x: (x.tdate - start_date).dt.days)

    df_transfers = pd.merge(left=df_transfers, right=df_visits, on=["vid", "pid"], how="left")
    series_time_diff = (df_transfers.tdate - df_transfers.adate)
    series_time_diff_in_days = (df_transfers.tdate - df_transfers.adate).dt.days

    df_transfers.insert(value=series_time_diff, column="tdiff", loc=df_transfers.shape[1])
    df_transfers.insert(value=series_time_diff_in_days, column="tdiff_days", loc=df_transfers.shape[1])

    pid_in_MICU_transfer = df_transfers.pid.unique()

    # Only take a look at transfers that occured at least 3 days since admission
    # transfer into MICU in less than 3 days are excluded from dataset
    df_transfers_true_instances = df_transfers[df_transfers.tdiff_days >= 3]
    df_transfers_remove_instances = df_transfers[df_transfers.tdiff_days < 3]

    df_MICU_transfer = df_transfers_true_instances[["pid", "timestamp"]] 
    df_MICU_transfer = df_MICU_transfer.rename(columns={"pid": "user_id"})
    # Sometimes, there are more than one record per patient. Only keep the latest
    df_MICU_transfer = df_MICU_transfer.sort_values(by="timestamp")
    df_MICU_transfer.drop_duplicates(subset="user_id", keep="last", inplace=True)
    df_MICU_transfer.insert(loc=df_MICU_transfer.shape[1], value=1, column="label")

    df_visits_nonMICU_transfer = df_visits[~df_visits["pid"].isin(pid_in_MICU_transfer)]

    # Generate negative instance by sampling a random date
    df_visits_nonMICU_transfer.insert(column="start", loc=4, value=start_date)
    df_visits_nonMICU_transfer.insert(column="end", loc=5, value=train_end_date -  pd.Timedelta(days=1))
    df_visits_nonMICU_transfer.insert(column="start_rand", loc=1, value=pd.to_datetime(df_visits_nonMICU_transfer[["adate", "start"]].max(axis=1).dt.date))
    df_visits_nonMICU_transfer.insert(column="end_rand", loc=2, value=pd.to_datetime(df_visits_nonMICU_transfer[["ddate", "end"]].min(axis=1).dt.date))
    df_visits_nonMICU_transfer.insert(column="days", loc=3, value=(df_visits_nonMICU_transfer.end_rand - df_visits_nonMICU_transfer.start_rand).dt.days)

    days_to_add = [random.randint(0, days) for days in df_visits_nonMICU_transfer.days.values]
    df_visits_nonMICU_transfer.insert(column="days_to_add", value=days_to_add, loc=3)

    df_visits_nonMICU_transfer.insert(column="date_timestamp", loc=1, value=pd.to_datetime((df_visits_nonMICU_transfer["start_rand"] + pd.to_timedelta(df_visits_nonMICU_transfer["days_to_add"], unit='d')).dt.date))
    ###################
    # get df_nonMICU_transfer
    df_visits_nonMICU_transfer = df_visits_nonMICU_transfer.rename(columns={"pid": "user_id"})
    df_nonMICU_transfer = df_visits_nonMICU_transfer[["user_id", "date_timestamp"]]
    df_nonMICU_transfer.insert(loc=df_nonMICU_transfer.shape[1], value=0, column="label")

    df_nonMICU_transfer.insert(column="start_date", value=start_date, loc=df_nonMICU_transfer.shape[1])
    df_nonMICU_transfer.insert(column="timestamp", loc=1, value=(df_nonMICU_transfer["date_timestamp"] - df_nonMICU_transfer["start_date"]).dt.days)
    df_nonMICU_transfer = df_nonMICU_transfer[["user_id", "timestamp", "label"]]

    df_MICU_transfer = pd.concat([df_MICU_transfer, df_nonMICU_transfer])
    return df_MICU_transfer

def save_datasets():
    df_mortality.to_csv("../data/patient_label/df_mortality.csv", index=False)
    df_severity.to_csv("../data/patient_label/df_severity.csv", index=False)
    df_CDI.to_csv("../data/patient_label/df_CDI.csv", index=False)
    df_MICU_transfer.to_csv("../data/patient_label/df_MICU_transfer.csv", index=False)

if __name__ == "__main__":
    start_date = pd.Timestamp(2010,1,1)
    train_end_date = pd.Timestamp(2010,4,1)
    start_date_str = start_date.strftime('%Y-%m-%d')
    train_end_date_str = train_end_date.strftime('%Y-%m-%d')

    df_visits = get_df_visits(start_date_str, train_end_date_str)

    print("df_mortality. df_severity")
    df_mortality, df_severity = gen_df_mortality_severity(df_visits)

    print("df_CDI")
    df_CDI = gen_df_CDI(df_visits)

    print("df_MICU_transfer")
    df_MICU_transfer = gen_df_MICU_transfer(df_visits)

    save_datasets()
