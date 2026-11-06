"""
WA (SWIS/WEM) day-ahead electricity demand forecast
---------------------------------------------------
Compares a LightGBM model against two simple baselines using a strict
chronological train/test split and only features that would be known at
forecast time (no leakage).

USAGE
  Real data (AEMO OperationalDemandWithdrawal files, comma-separated or wildcard):
    python wa_demand_forecast.py --demand-csv "OperationalDemandWithdrawal-*.csv" --test-months 12
  Pipeline test on synthetic data (NOT real results):
    python wa_demand_forecast.py --demo

The demand CSV needs a timestamp column and a demand (MW) column. Check the
header names in the file you download and pass them with --time-col and
--demand-col. Weather (Perth hourly temperature) is fetched automatically
from Open-Meteo unless you pass --weather-csv.
"""
import argparse
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PERTH_LAT, PERTH_LON = -31.95, 115.86
HORIZON_H = 24  # forecast 24 hours ahead


# ---------------------------------------------------------------- data loading
def load_demand(path, time_col, demand_col):
    import glob
    files = []
    for part in path.split(","):
        files += sorted(glob.glob(part.strip())) or [part.strip()]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(files)} file(s), {len(df):,} rows")
    if time_col not in df.columns or demand_col not in df.columns:
        sys.exit(f"Columns not found. Available columns: {list(df.columns)}")
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[time_col, demand_col])
    s = df.set_index(time_col)[demand_col].astype(float).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    # aggregate to hourly mean so it aligns with hourly weather
    return s.resample("h").mean().interpolate(limit=3).rename("demand")


def fetch_weather(start, end):
    import requests

    url = "https://archive-api.open-meteo.com/v1/archive"
    end = min(end, pd.Timestamp.today().normalize() - pd.Timedelta(days=7))
    params = dict(
        latitude=PERTH_LAT,
        longitude=PERTH_LON,
        start_date=str(start.date()),
        end_date=str(end.date()),
        hourly="temperature_2m",
        timezone="Australia/Perth",
    )
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    w = pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]), name="temp")
    return w


def make_demo():
    """Synthetic demand + temperature, only to test the pipeline."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-11-01", "2025-10-31 23:00", freq="h")
    doy = idx.dayofyear.values
    temp = (
        18 + 8 * np.cos(2 * np.pi * (doy - 30) / 365)
        + 5 * np.sin(2 * np.pi * (idx.hour.values - 9) / 24)
        + rng.normal(0, 2, len(idx))
    )
    cool = np.maximum(temp - 24, 0)
    heat = np.maximum(15 - temp, 0)
    daily = 1800 + 500 * np.exp(-((idx.hour.values - 18) ** 2) / 12)
    wk = np.where(idx.dayofweek.values >= 5, -150, 0)
    demand = daily + wk + 60 * cool + 25 * heat + rng.normal(0, 40, len(idx))
    return (pd.Series(demand, idx, name="demand"),
            pd.Series(temp, idx, name="temp"))


# ------------------------------------------------------------------- features
def build_features(d, w):
    """
    Target at time T = demand at T. Forecast is issued at T-24h, so every
    feature must come from T-24h or earlier, EXCEPT calendar fields and
    temperature at T (in practice a weather forecast; see README note).
    """
    df = pd.concat([d, w], axis=1).dropna()
    out = pd.DataFrame(index=df.index)
    out["y"] = df["demand"]
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    out["month"] = df.index.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["temp"] = df["temp"]                       # temperature at target time
    out["temp_max_prev24"] = df["temp"].rolling(24).max().shift(HORIZON_H)
    # lags that are available 24h before the target
    out["lag_24"] = df["demand"].shift(24)
    out["lag_48"] = df["demand"].shift(48)
    out["lag_168"] = df["demand"].shift(168)
    out["roll_mean_24_lagged"] = df["demand"].rolling(24).mean().shift(HORIZON_H)
    return out.dropna()


# ----------------------------------------------------------------- evaluation
def metrics(y, p):
    err = y - p
    return dict(MAE=np.mean(np.abs(err)),
                RMSE=np.sqrt(np.mean(err ** 2)),
                MAPE=np.mean(np.abs(err / y)) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand-csv")
    ap.add_argument("--time-col", default="Dispatch Interval")
    ap.add_argument("--demand-col", default="Operational Demand")
    ap.add_argument("--weather-csv", help="optional CSV with columns time,temp")
    ap.add_argument("--demo", action="store_true", help="synthetic data, pipeline test only")
    ap.add_argument("--test-months", type=int, default=6)
    a = ap.parse_args()

    if a.demo:
        d, w = make_demo()
        print("*** DEMO MODE: synthetic data. Do NOT quote these numbers anywhere. ***\n")
    else:
        if not (a.demand_csv and a.time_col and a.demand_col):
            sys.exit("Provide --demand-csv, --time-col and --demand-col (or use --demo).")
        d = load_demand(a.demand_csv, a.time_col, a.demand_col)
        if a.weather_csv:
            wdf = pd.read_csv(a.weather_csv, parse_dates=["time"]).set_index("time")
            w = wdf["temp"]
        else:
            print("Fetching Perth hourly temperature from Open-Meteo...")
            w = fetch_weather(d.index.min(), d.index.max())
        # make both tz-naive local time
        d.index = d.index.tz_localize(None) if d.index.tz else d.index
        w.index = w.index.tz_localize(None) if w.index.tz else w.index

    data = build_features(d, w)
    n_hours = len(data)

    # chronological split: last N months held out
    cutoff = data.index.max() - pd.DateOffset(months=a.test_months)
    train, test = data[data.index <= cutoff], data[data.index > cutoff]
    # inner validation (last 10% of train) for early stopping, still chronological
    v_cut = int(len(train) * 0.9)
    tr, va = train.iloc[:v_cut], train.iloc[v_cut:]
    X_cols = [c for c in data.columns if c != "y"]

    print(f"Observations (hourly): {n_hours:,}")
    print(f"Train: {train.index.min().date()} to {train.index.max().date()}  ({len(train):,})")
    print(f"Test : {test.index.min().date()} to {test.index.max().date()}  ({len(test):,})\n")

    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(tr[X_cols], tr["y"], eval_set=[(va[X_cols], va["y"])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = model.predict(test[X_cols])

    base_day = test["lag_24"].values      # persistence: same hour yesterday
    base_week = test["lag_168"].values    # seasonal naive: same hour last week

    rows = {
        "Persistence (same hour yesterday)": metrics(test["y"].values, base_day),
        "Seasonal naive (same hour last week)": metrics(test["y"].values, base_week),
        "LightGBM": metrics(test["y"].values, pred),
    }
    res = pd.DataFrame(rows).T.round(2)
    print(res.to_string(), "\n")

    best_base = min(rows[k]["MAE"] for k in list(rows)[:2])
    imp = (best_base - rows["LightGBM"]["MAE"]) / best_base * 100
    print(f"MAE reduction vs best baseline: {imp:.1f}%")
    print(f"MAE reduction vs persistence  : "
          f"{(rows['Persistence (same hour yesterday)']['MAE'] - rows['LightGBM']['MAE']) / rows['Persistence (same hour yesterday)']['MAE'] * 100:.1f}%\n")

    fi = pd.Series(model.feature_importances_, X_cols).sort_values()
    res.to_csv("results.csv")

    # ---- charts
    last = test.iloc[: 24 * 14]
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    ax[0].plot(last.index, last["y"], label="Actual", lw=1.6)
    ax[0].plot(last.index, pred[: len(last)], label="LightGBM", lw=1.2)
    ax[0].plot(last.index, base_day[: len(last)], label="Persistence", lw=1, alpha=0.6)
    ax[0].set_title("Day-ahead demand forecast: first 2 weeks of test set")
    ax[0].set_ylabel("Demand (MW)")
    ax[0].legend()
    ax[1].barh(fi.index, fi.values)
    ax[1].set_title("LightGBM feature importance (split count)")
    plt.tight_layout()
    plt.savefig("forecast_chart.png", dpi=150)
    print("Saved results.csv and forecast_chart.png")


if __name__ == "__main__":
    main()
