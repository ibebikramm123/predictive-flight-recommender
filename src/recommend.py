import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Load trained model once
# -----------------------------
MODEL_PATH = Path("models/model.joblib")
PIPE = joblib.load(MODEL_PATH)


def _norm_airline(x: str) -> str:
    """
    Normalize airline strings like:
    'Delta||Delta' -> 'Delta'
    'United||United' -> 'United'
    'American Airlines||American Airlines' -> 'American Airlines'
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if "||" in s:
        s = s.split("||")[0].strip()
    return s


# --------------------------------------------------
# 1) FILTER CANDIDATES (Future-Date Compatible)
# --------------------------------------------------
def filter_candidates(df: pd.DataFrame, origin: str, dest: str, flight_date: str) -> pd.DataFrame:
    # Filter only by route (NOT by historical date)
    route_df = df[
        (df["startingAirport"].astype(str).str.upper() == origin.upper()) &
        (df["destinationAirport"].astype(str).str.upper() == dest.upper())
    ].copy()

    if route_df.empty:
        return route_df

    user_date = pd.to_datetime(flight_date, errors="coerce")
    if pd.isna(user_date):
        return route_df.iloc[0:0].copy()  # invalid date -> empty

    today = pd.Timestamp.today().normalize()

    # Recompute dynamic date features
    route_df["flightDate"] = user_date
    route_df["days_to_departure"] = (user_date - today).days
    route_df["days_to_departure"] = route_df["days_to_departure"].clip(lower=0)
    route_df["dow"] = int(user_date.dayofweek)
    route_df["month"] = int(user_date.month)

    # Normalize airline name for grouping + preference matching
    if "segmentsAirlineName" in route_df.columns:
        route_df["airline_clean"] = route_df["segmentsAirlineName"].apply(_norm_airline)
    else:
        route_df["airline_clean"] = ""

    # Optional speed optimization
    if len(route_df) > 5000:
        route_df = route_df.sample(n=5000, random_state=42)

    return route_df


# --------------------------------------------------
# 2) RECOMMEND FUNCTION
# --------------------------------------------------
def recommend(
    df: pd.DataFrame,
    mode: str = "Cheapest",
    preferred_airlines=None,
    topk: int = 20,
    max_per_airline: int = 1
) -> pd.DataFrame:

    if df is None or df.empty:
        return df

    work = df.copy()

    # Ensure airline_clean exists
    if "airline_clean" not in work.columns:
        if "segmentsAirlineName" in work.columns:
            work["airline_clean"] = work["segmentsAirlineName"].apply(_norm_airline)
        else:
            work["airline_clean"] = ""

    # Preferred airlines normalize
    preferred = []
    if preferred_airlines:
        preferred = [_norm_airline(a) for a in preferred_airlines if str(a).strip()]

    # Predict (pipeline will select required columns internally)
    work["pred_price"] = PIPE.predict(work)

    # Bonus if preferred airline
    work["airline_bonus"] = work["airline_clean"].apply(lambda a: 1 if a in preferred else 0)

    mode = (mode or "Cheapest").strip().lower()

    # Ranking logic
    if mode == "cheapest":
        work = work.sort_values(["pred_price", "duration_mins", "stops_est"], ascending=[True, True, True])

    elif mode == "fastest":
        # fastest, but keep price sensible second
        work = work.sort_values(["duration_mins", "pred_price", "stops_est"], ascending=[True, True, True])

    else:
        # recommended score: lower is better
        # (tune weights any time)
        work["score"] = (
            work["pred_price"] * 0.60 +
            work["duration_mins"] * 0.30 +
            work["stops_est"] * 50.0 -
            work["airline_bonus"] * 120.0
        )
        work = work.sort_values(["score", "pred_price"], ascending=[True, True])

    # Remove exact duplicates (helpful if dataset has repeated rows)
    work = work.drop_duplicates()

    # Force multiple airlines in results (avoid one airline dominating)
    if max_per_airline and "airline_clean" in work.columns:
        work = work.groupby("airline_clean", group_keys=False).head(int(max_per_airline))

    # Add simple tag
    med = float(work["pred_price"].median())
    work["tag"] = work["pred_price"].apply(lambda x: "Good deal" if float(x) <= med else "")

    return work.head(int(topk)).copy()