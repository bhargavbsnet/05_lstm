"""
data_prep.py
------------
Shared data preparation for every model script in this folder.

It does the same job as Steps 1 to 6 of the main notebook:
  raw receipts  ->  customers per hour  ->  features  ->  train / test split

Every model script starts with:

    from data_prep import load_data
    train, test, FEATURES, TARGET = load_data()

so that all models are trained and tested on exactly the same data.
"""

import glob
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
OPEN_HOUR = 7          # first trading hour
CLOSE_HOUR = 19        # last trading hour
SPLIT_DATE = pd.Timestamp("2022-07-01")   # training before, testing after

HOLIDAYS = pd.to_datetime([
    "2021-01-01", "2021-04-05", "2021-05-01", "2021-05-08", "2021-05-13", "2021-05-24",
    "2021-07-14", "2021-08-15", "2021-11-01", "2021-11-11", "2021-12-25",
    "2022-01-01", "2022-04-18", "2022-05-01", "2022-05-08", "2022-05-26",
    "2022-06-06", "2022-07-14", "2022-08-15",
])

FEATURES = ["hour", "day_number", "month", "is_weekend", "is_holiday",
            "last_week", "two_weeks_ago", "recent_average"]
TARGET = "customers"


def find_csv():
    """Locate the Kaggle sales file, or offer an upload box in Google Colab."""
    files = glob.glob("*akery*sales*.csv") + glob.glob("data/*akery*sales*.csv")
    if files:
        return files[0]
    try:
        from google.colab import files as colab_files
        uploaded = colab_files.upload()
        return list(uploaded.keys())[0]
    except Exception:
        raise FileNotFoundError(
            "Put 'Bakery sales.csv' next to this script, or download it from "
            "https://www.kaggle.com/datasets/matthieugimbert/french-bakery-daily-sales"
        )


def load_data(verbose=True):
    """Return train, test, FEATURES, TARGET ready for modelling."""
    sales = pd.read_csv(find_csv())
    sales = sales.loc[:, ~sales.columns.str.startswith("Unnamed")]

    # --- clean: drop refunds and placeholder products -----------------
    sales["timestamp"] = pd.to_datetime(sales["date"] + " " + sales["time"])
    sales = sales[sales["Quantity"] > 0]
    sales = sales[sales["article"].str.strip() != "."]

    # --- one receipt = one customer -----------------------------------
    sales["hour_slot"] = sales["timestamp"].dt.floor("h")
    hourly = sales.groupby("hour_slot")["ticket_number"].nunique().rename("customers")

    # --- rebuild the full calendar of trading hours, zeros included ----
    all_days = pd.date_range(hourly.index.min().normalize(),
                             hourly.index.max().normalize(), freq="D")
    slots = pd.DatetimeIndex([day + pd.Timedelta(hours=h)
                              for day in all_days
                              for h in range(OPEN_HOUR, CLOSE_HOUR + 1)])
    demand = hourly.reindex(slots, fill_value=0).to_frame()
    demand.index.name = "datetime"

    # --- days with no sales at all mean the shop was closed ------------
    per_day = demand["customers"].groupby(demand.index.normalize()).transform("sum")
    demand["is_open_day"] = (per_day > 0).astype(int)

    # --- calendar features ---------------------------------------------
    demand["hour"] = demand.index.hour
    demand["weekday"] = demand.index.day_name()
    demand["day_number"] = demand.index.dayofweek
    demand["month"] = demand.index.month
    demand["is_weekend"] = (demand["day_number"] >= 5).astype(int)
    demand["is_holiday"] = demand.index.normalize().isin(HOLIDAYS).astype(int)

    # --- past-demand features (never shorter than 7 days) --------------
    same_hour = demand.groupby("hour")["customers"]
    demand["last_week"] = same_hour.shift(7)
    demand["two_weeks_ago"] = same_hour.shift(14)
    demand["recent_average"] = same_hour.transform(
        lambda s: s.shift(7).rolling(7, min_periods=3).mean())

    model_data = demand[demand["is_open_day"] == 1].dropna(subset=FEATURES)

    train = model_data[model_data.index < SPLIT_DATE]
    test = model_data[model_data.index >= SPLIT_DATE]

    if verbose:
        print(f"Training hours: {len(train):,}  "
              f"({train.index.min().date()} to {train.index.max().date()})")
        print(f"Testing hours:  {len(test):,}  "
              f"({test.index.min().date()} to {test.index.max().date()})")
        print()

    return train, test, FEATURES, TARGET


def report(model_name, actual, predicted, save_plot=True):
    """Print the score for one model and optionally save a one-week chart."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print("-" * 52)
    print(f"MODEL: {model_name}")
    print(f"  MAE  = {mae:.2f} customers per hour  (lower is better)")
    print(f"  RMSE = {rmse:.2f}")
    print("-" * 52)

    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("outputs", exist_ok=True)
        hours = range(min(91, len(actual)))          # first week of the test period
        plt.figure(figsize=(13, 4))
        plt.plot(hours, np.asarray(actual)[:len(hours)], label="Actual", linewidth=2)
        plt.plot(hours, predicted[:len(hours)], label="Predicted", linewidth=2, linestyle="--")
        plt.title(f"{model_name}: first week of the test period (MAE {mae:.2f})")
        plt.ylabel("customers per hour")
        plt.legend()
        plt.tight_layout()
        name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f"outputs/{name}.png", dpi=140)
        plt.close()
        print(f"Chart saved to outputs/{name}.png\n")

    return mae
