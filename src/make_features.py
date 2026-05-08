import polars as pl

RAW = "data/raw/itineraries.csv"
OUT = "data/processed/features.parquet"

columns = [
    "legId",
    "searchDate",
    "flightDate",
    "startingAirport",
    "destinationAirport",
    "fareBasisCode",
    "travelDuration",
    "elapsedDays",
    "isBasicEconomy",
    "isRefundable",
    "isNonStop",
    "baseFare",
    "totalFare",
    "seatsRemaining",
    "totalTravelDistance",
    "departureTimeEpoch",
    "departureTime",
    "arrivalTimeEpoch",
    "arrivalTime",
    "returnStartingAirport",
    "returnDestinationAirport",
    "segmentsAirlineName",
    "segmentsAirlineCode",
    "segmentsEquipmentDescription",
    "segmentsDurationInSeconds",
    "segmentsDistance",
    "segmentsCabinCode",
]

def iso8601_to_minutes(expr: pl.Expr) -> pl.Expr:
    h = expr.str.extract(r"(\d+)H", 1).cast(pl.Int64, strict=False).fill_null(0)
    m = expr.str.extract(r"(\d+)M", 1).cast(pl.Int64, strict=False).fill_null(0)
    return (h * 60 + m).alias("duration_mins")

print("Reading dataset...")
lf = pl.scan_csv(RAW, has_header=False, skip_rows=1)
lf = lf.rename(dict(zip(lf.columns, columns)))

df = (
    lf.with_columns([
        pl.col("searchDate").str.strptime(pl.Date, "%d/%m/%Y", strict=False),
        pl.col("flightDate").str.strptime(pl.Date, "%d/%m/%Y", strict=False),
        pl.col("segmentsAirlineName")
        .cast(pl.Utf8)
        .str.split("||")
        .list.first()
        .alias("segmentsAirlineName"),
    ])
    .with_columns([
        (pl.col("flightDate") - pl.col("searchDate")).dt.total_days().alias("days_to_departure"),
        pl.col("flightDate").dt.weekday().alias("dow"),
        pl.col("flightDate").dt.month().alias("month"),
        iso8601_to_minutes(pl.col("travelDuration")),
        pl.col("segmentsAirlineCode").str.count_matches(r"\|").alias("stops_est"),
    ])
    .select([
        "startingAirport", "destinationAirport",
        "searchDate", "flightDate",
        "days_to_departure", "dow", "month",
        "duration_mins", "stops_est",
        "isBasicEconomy", "isRefundable", "isNonStop",
        "seatsRemaining", "totalTravelDistance",
        "segmentsAirlineName", "segmentsCabinCode",
        "departureTime", "arrivalTime",
        "totalFare"
    ])
    .filter(pl.col("totalFare").is_not_null())
)

print("Saving processed data...")
df.collect(engine="streaming").write_parquet(OUT)
print(" Features created at:", OUT)