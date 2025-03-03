# Data Preparation for Granite-TSFM Retail Example

#


import logging
import os
from itertools import chain

import numpy as np
import pandas as pd


logger = logging.getLogger(__file__)

try:
    import gdown
except ImportError:
    logger.error("Please install the `gdown` utility to enable downloading the data (`pip install gdown`).")


def prepare_data(temp_dir: str = "temp/", force_download: bool = False):
    """Utility to prepare the state-level aggregated M5 dataset for use with granite-timeseries-ttm retail example

    Args:
        temp_dir (str): A folder for the temporary downloaded files, defaults to "temp/".
        force_download (bool): If True data will be downloaded again. Default False.

    Download the M5 datasets from the google drive mentioned on the official M-Competitions
    [repository](https://github.com/Mcompetitions/M5-methods). The google drive containing all datasets and
    additional competition information is
    [here](https://drive.google.com/drive/folders/1D6EWdVSaOtrP1LEFh1REjI3vej6iUS_4?usp=sharing). The required files
    are downloaded to a folder named temp, once the data is prepared, it is safe to delete this folder.
    """

    if os.path.exists(temp_dir) and not force_download:
        logger.info("Temporary folder already exists, assuming data already downloaded.")
    else:
        gdown.download(id="1Bj7Xj15yn4j6-BM_mpSmOVTV0c2ihEdo", output=temp_dir)
        gdown.download(id="1-khP8tp1gRTfaQQV_PlKd6R_3Mw5KiBw", output=temp_dir)
        gdown.download(id="1eeU8Me44yzgWsDD0dYrX4xVOixil_H4u", output=temp_dir)

    df_calendar = pd.read_csv(os.path.join(temp_dir, "calendar.csv"))
    df_sales_train = pd.read_csv(os.path.join(temp_dir, "sales_train_evaluation.csv"))
    df_sales_test = pd.read_csv(os.path.join(temp_dir, "sales_test_evaluation.csv"))

    # add missing "d" column for later merging
    df_calendar["d"] = (
        (pd.to_datetime(df_calendar["date"]) - pd.to_datetime(df_calendar["date"]).min()).dt.days + 1
    ).transform(lambda x: f"d_{x}")

    # Prepare Data
    """
    The data is contained in two core files `sales_train_evaluation.csv` and `sales_test_evaluation.csv`. In each of
    these files, the rows represent complete time series for individual store-item combinations. For use with
    granite-timeseries-ttm, we need a single dataset where each row represents a single timestamp for a store-item
    combination. In addition, since we will be looking at the sales only at the combined state level sales, we need to
    aggregate by the state id. The overall data preparation strategy is as follows:
    1. Aggregate the individual data frames by `state_id`.
    2. Melt the data frames to transform the data to "long format", where each row has a timestamp
    3. Concatenate the train and test into one dataset
    4. Merge the calendar features from `calendar.csv`
    5. Encode date features into numerical quantities
    6. Add group statistics to the data
    7. Format categorical columns

    After the data processing, we end up with a single dataframe containg three state-level time series.
    """

    # Aggregate
    ID_VARS = ["state_id"]
    DATE_FEATURES = [
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]

    df_sales_train = (
        df_sales_train.groupby("state_id")
        .sum()
        .drop(columns=["item_id", "dept_id", "cat_id", "store_id"])
        .reset_index()
    )
    df_sales_train = df_sales_train.melt(id_vars=ID_VARS, var_name="d", value_name="sales")

    df_sales_test = (
        df_sales_test.groupby("state_id")
        .sum()
        .drop(columns=["item_id", "dept_id", "cat_id", "store_id"])
        .reset_index()
    )
    df_sales_test = df_sales_test.melt(id_vars=ID_VARS, var_name="d", value_name="sales")
    df_all = pd.concat([df_sales_train, df_sales_test]).reset_index(drop=True)
    df_all = pd.merge(df_all, df_calendar[["d"] + DATE_FEATURES], on="d", how="left")

    # Encode date features using sin-cos embedding
    date_cols = [
        "wm_yr_wk",
        "wday",
        "month",
    ]
    for k in date_cols:
        unq_nums = df_all[k].unique()
        period = len(unq_nums)
        # normalize
        df_all[k] = df_all[k] - df_all[k].min()
        # get encodings
        df_all[k + "_sin"] = np.sin(2 * np.pi * df_all[k] / period)

    # Add grouped statistics as used in lightGBM by winners
    df_tmp = df_all[["date", "d", "state_id", "sales"]].copy()
    df_tmp["idx"] = pd.to_datetime(df_tmp["date"])
    df_tmp["idx"] = (df_tmp["idx"] - df_tmp["idx"].min()).dt.days + 1
    # mask out the sales during the test period
    df_tmp.loc[df_tmp["idx"] > (1941 - 28), "sales"] = np.nan

    icols = [
        ["state_id"],
    ]

    for col in icols:
        col_name = "_" + "_".join(col) + "_"
        df_tmp["enc" + col_name + "mean"] = df_tmp.groupby(col)["sales"].transform("mean")
        df_tmp["enc" + col_name + "std"] = df_tmp.groupby(col)["sales"].transform("std")

    encoding_cols = [col for col in df_tmp.columns if col not in ["d", "date", "idx", "sales"]]
    df_all = pd.merge(df_all, df_tmp[["d"] + encoding_cols], on=["d"] + list(chain(*icols)), how="left")

    # Format Categorical Columns
    categoricals = [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]

    df_all[categoricals] = df_all[categoricals].fillna("noevent")

    # create a copy of state_id for use as a categorical
    df_all["state_id_cat"] = df_all["state_id"]

    outfile = "m5_for_state_level_forecasting.csv.gz"
    df_all.to_csv(outfile, index=False)
    print(f"Successfully saved the prepared M5 data to {outfile}.")
