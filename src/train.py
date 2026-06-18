import polars as pl
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

DATA = "data/processed/features.parquet"
MODEL_OUT = "models/model.joblib"

def main():
    df = pl.read_parquet(DATA).to_pandas()

    TARGET = "totalFare"

    feature_cols = [
        "startingAirport", "destinationAirport", "segmentsAirlineName", "segmentsCabinCode",
        "days_to_departure", "dow", "month",
        "duration_mins", "stops_est",
        "seatsRemaining", "totalTravelDistance",
        "isBasicEconomy", "isRefundable", "isNonStop",
    ]

    # keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    missing = [c for c in ["dow", "month", "duration_mins", "stops_est"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered columns {missing}. Run make_features.py again.")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")

    # drop rows where target is missing
    X = X.loc[y.notna()].copy()
    y = y.loc[y.notna()].copy()

    # bool -> 0/1 (important)
    for c in ["isBasicEconomy", "isRefundable", "isNonStop"]:
        if c in X.columns:
            X[c] = X[c].astype(int)

    cat_cols = ["startingAirport", "destinationAirport", "segmentsAirlineName", "segmentsCabinCode"]
    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # ensure numeric columns are numeric
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
        ]), num_cols),
    ])

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    joblib.dump(pipe, MODEL_OUT)
    print("Saved model to:", MODEL_OUT)

if __name__ == "__main__":
    main()
    