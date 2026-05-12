


import os
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify, render_template
import polars as pl
import airportsdata

from src.recommend import filter_candidates, recommend

app = Flask(__name__, template_folder="templates", static_folder="static")

DATA = ROOT / "data/processed/features.parquet"

# Load once at startup
DF = pl.read_parquet(DATA).to_pandas()

try:
    airports_data = airportsdata.load('IATA')
except Exception as e:
    print(f"Failed to load airports data: {e}")
    airports_data = {}


def apply_stops_filter(df, stops_pref: str):
    """stops_pref: '0','1','2','3' where 3 means 2+ stops"""
    if df is None or df.empty:
        return df

    if stops_pref is None or stops_pref == "":
        return df

    if "stops_est" not in df.columns:
        return df

    try:
        s = int(stops_pref)
    except ValueError:
        return df

    if s >= 3:
        return df[df["stops_est"] >= 2]

    return df[df["stops_est"] == s]


def add_time_aliases(df):
    """
    Adds standard frontend-friendly columns:
    departure_time
    arrival_time

    If real time columns do not exist, it creates empty values instead of crashing.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    departure_candidates = [
        "segmentsDepartureTimeRaw",
        "segmentsDepartureTime",
        "legDepartureTime",
        "departure_time",
        "departureTime",
        "searchDate",
    ]

    arrival_candidates = [
        "segmentsArrivalTimeRaw",
        "segmentsArrivalTime",
        "legArrivalTime",
        "arrival_time",
        "arrivalTime",
    ]

    dep_col = next((c for c in departure_candidates if c in df.columns), None)
    arr_col = next((c for c in arrival_candidates if c in df.columns), None)

    df["departure_time"] = df[dep_col] if dep_col else ""
    df["arrival_time"] = df[arr_col] if arr_col else ""

    return df


def clean_records(records):
    """Convert NaN/inf + timestamp-like values into JSON-safe Python types."""
    cleaned = []

    for r in records:
        rr = {}

        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                rr[k] = None
            elif hasattr(v, "isoformat"):
                rr[k] = v.isoformat()
            else:
                rr[k] = v

        cleaned.append(rr)

    return cleaned


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/airports")
def get_airports():
    airports_data = airportsdata.load('IATA')
    codes = sorted(
        set(DF["startingAirport"].unique()).union(
            set(DF["destinationAirport"].unique())
        )
    )
    airports = []
    for code in codes:
        info = airports_data.get(code.upper(), {})
        name = info.get('name', code)
        airports.append({"code": code, "name": name})
    return jsonify(airports)


@app.get("/api/insights")
def get_insights():
    origin = request.args.get('origin', '').strip().upper()
    dest = request.args.get('dest', '').strip().upper()
    
    if not origin or not dest:
        return jsonify({"insights": "Please provide valid origin and destination airports."})
    
    # Filter dataset for the route
    subset = DF[(DF['startingAirport'] == origin) & (DF['destinationAirport'] == dest)]
    
    if subset.empty:
        return jsonify({"insights": f"No historical data available for flights from {origin} to {dest}."})
    
    # Calculate insights
    avg_price = subset['totalFare'].mean()
    min_price = subset['totalFare'].min()
    max_price = subset['totalFare'].max()
    popular_airline = subset['segmentsAirlineName'].mode().iloc[0] if not subset['segmentsAirlineName'].mode().empty else 'Unknown'
    avg_duration = subset['duration_mins'].mean()
    avg_stops = subset['stops_est'].mean()
    total_flights = len(subset)
    avg_distance = subset['totalTravelDistance'].mean()
    co2_per_km = 0.09  # Rough estimate for economy flight CO2 emissions in kg per km per passenger
    avg_co2 = avg_distance * co2_per_km if not math.isnan(avg_distance) else 0
    
    # Price by day of week
    price_by_dow = subset.groupby('dow')['totalFare'].mean()
    best_dow = price_by_dow.idxmin()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    best_day = days[int(best_dow)]
    
    insights = f"""
    🤖 AI Travel Insights for {origin} → {dest}:
    
    📊 Data from {total_flights} historical flights
    💰 Average price: ${avg_price:.2f} (Range: ${min_price:.2f} - ${max_price:.2f})
    ✈️ Most popular airline: {popular_airline}
    ⏱️ Average flight duration: {int(avg_duration // 60)}h {int(avg_duration % 60)}m
    🛑 Average stops: {avg_stops:.1f}
    🌍 Average distance: {avg_distance:.0f} km
    🌱 Estimated CO2 emissions: {avg_co2:.0f} kg per passenger (economy class)
    📅 Best day to fly: {best_day} (lowest average price)
    
    ⚠️ Accuracy Note: Insights based on 2022 historical data. Actual prices and conditions may vary due to market changes, fuel costs, and current events.
    
    💡 Tip: Book 2-3 weeks in advance for best prices. Consider {popular_airline} for reliability. For eco-friendly travel, look for direct flights.
    """
    
    return jsonify({"insights": insights.strip()})


@app.post("/api/recommend")
def api_recommend():
    p = request.get_json(force=True) or {}

    origin = (p.get("from") or "").strip().upper()
    dest = (p.get("to") or "").strip().upper()
    depart_date = (p.get("departDate") or "").strip()
    return_date = (p.get("returnDate") or "").strip()
    mode = (p.get("mode") or "Cheapest").strip()
    max_price = float(p.get("maxPrice", 1e9))
    stops = str(p.get("stops", "")).strip()

    preferred_airlines = p.get("preferredAirlines", [])
    preferred_airlines = [
        str(a).strip()
        for a in preferred_airlines
        if str(a).strip()
    ]

    if not origin or not dest or not depart_date:
        return jsonify({
            "error": "Missing required fields: from, to, departDate"
        }), 400

    keep_cols = [
        "segmentsAirlineName",
        "segmentsCabinCode",
        "duration_mins",
        "stops_est",
        "seatsRemaining",
        "pred_price",
        "tag",
        "departure_time",
        "arrival_time",
    ]

    # -----------------------------
    # OUTBOUND
    # -----------------------------
    cand_out = filter_candidates(DF, origin, dest, depart_date)
    cand_out = apply_stops_filter(cand_out, stops)

    out_ranked = recommend(
        cand_out,
        mode=mode,
        preferred_airlines=preferred_airlines,
        topk=200,
    )

    out_ranked = add_time_aliases(out_ranked)

    if not out_ranked.empty and "pred_price" in out_ranked.columns:
        out_ranked = out_ranked[out_ranked["pred_price"] <= max_price]

    keep_cols_out = [
        c for c in keep_cols
        if not out_ranked.empty and c in out_ranked.columns
    ]

    outbound = (
        out_ranked[keep_cols_out].head(10).to_dict(orient="records")
        if not out_ranked.empty
        else []
    )

    # -----------------------------
    # RETURN
    # -----------------------------
    inbound = []

    if return_date:
        cand_ret = filter_candidates(DF, dest, origin, return_date)
        cand_ret = apply_stops_filter(cand_ret, stops)

        ret_ranked = recommend(
            cand_ret,
            mode=mode,
            preferred_airlines=preferred_airlines,
            topk=200,
        )

        ret_ranked = add_time_aliases(ret_ranked)

        if not ret_ranked.empty and "pred_price" in ret_ranked.columns:
            ret_ranked = ret_ranked[ret_ranked["pred_price"] <= max_price]

        keep_cols_ret = [
            c for c in keep_cols
            if not ret_ranked.empty and c in ret_ranked.columns
        ]

        inbound = (
            ret_ranked[keep_cols_ret].head(10).to_dict(orient="records")
            if not ret_ranked.empty
            else []
        )

    return jsonify({
        "outbound": clean_records(outbound),
        "inbound": clean_records(inbound),
    })

@app.post("/api/chat")
def chat():
    data = request.get_json() or {}
    message = data.get('message', '').strip().lower()
    context = data.get('context', {})  # e.g., current route
    
    # Simple rule-based responses
    responses = {
        'hello': 'Hi! How can I help with your flight search today?',
        'hi': 'Hello! Ready to find the perfect flight?',
        'help': 'I can help with flight recommendations, price trends, booking tips, and travel advice. What would you like to know?',
        'best time': 'Generally, booking 2-3 weeks in advance gives the best prices, but check the price trend chart for your specific route.',
        'cheap': 'For cheaper flights, look for mid-week departures, be flexible with dates, and consider nearby airports.',
        'delay': 'Flight delays can happen due to weather, air traffic, or mechanical issues. Check airline apps for real-time updates.',
        'cancel': 'If you need to cancel, check the airline\'s policy. Some fares are non-refundable.',
        'baggage': 'Carry-on is usually free, but checked bags cost extra. Check airline policies.',
        'visa': 'Visa requirements vary by destination. Check the embassy website for your nationality.',
        'weather': 'Weather can affect flights. Check forecasts and consider flexible booking.',
        'default': 'I\'m here to help with flight planning! Ask me about prices, booking tips, or travel advice.'
    }
    
    response = responses.get(message, responses['default'])
    
    # If context has route, add personalized info
    if context.get('from') and context.get('to'):
        route = f"{context['from']} to {context['to']}"
        response += f" For your route {route}, I recommend checking the AI insights for personalized recommendations."
    
    return jsonify({"response": response})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
