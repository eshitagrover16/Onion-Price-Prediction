from flask import Flask, render_template, request
import joblib
import pandas as pd
import requests
import os

app = Flask(__name__)

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "onion_price_model.pkl")
FESTIVAL_PATH = os.path.join(BASE_DIR, "festival_calendar.xlsx")
DATA_PATH = os.path.join(BASE_DIR, "hp_onion_full_kaggle.csv")


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print("Error loading model:", e)


# ============================================================
# LOAD FESTIVAL DATA
# ============================================================

try:
    festival_df = pd.read_excel(FESTIVAL_PATH)

    # Convert date column to datetime
    festival_df["date"] = pd.to_datetime(
        festival_df["date"],
        errors="coerce"
    )

    festival_df = festival_df.dropna(subset=["date"])

    print("Festival calendar loaded successfully.")

except Exception as e:
    festival_df = pd.DataFrame()
    print("Error loading festival calendar:", e)


# ============================================================
# LOAD HISTORICAL DATASET
# ============================================================

try:
    historical_df = pd.read_csv(DATA_PATH)

    historical_df["date"] = pd.to_datetime(
        historical_df["date"],
        errors="coerce"
    )

    print("Historical dataset loaded successfully.")

except Exception as e:
    historical_df = pd.DataFrame()
    print("Error loading historical dataset:", e)


# ============================================================
# MARKET COORDINATES
# ============================================================

market_coordinates = {
    "Bilaspur": (31.3333, 76.7500),
    "Chamba": (32.5534, 76.1258),
    "Hamirpur": (31.6862, 76.5213),
    "Kangra": (32.0998, 76.2691),
    "Kullu": (31.9579, 77.1095),
    "Mandi": (31.7082, 76.9314),
    "Shimla": (31.1048, 77.1734),
    "Sirmaur": (30.5626, 77.2972),
    "Solan": (30.9045, 77.0967),
    "Una": (31.4685, 76.2708),
    "Kinnaur": (31.5833, 78.4167),
    "Lahaul and Spiti": (32.5717, 77.5369),
    "Mandi": (31.7082, 76.9314),
    "Kangra": (32.0998, 76.2691),
    "Kullu": (31.9579, 77.1095),
    "Hamirpur": (31.6862, 76.5213),
    "Bilaspur": (31.3333, 76.7500),
    "Chamba": (32.5534, 76.1258),
    "Solan": (30.9045, 77.0967),
    "Sirmaur": (30.5626, 77.2972),
    "Shimla": (31.1048, 77.1734),
    "Una": (31.4685, 76.2708),
    "Kinnaur": (31.5833, 78.4167),
    "Lahaul": (32.5717, 77.5369),
    "Spiti": (32.2467, 78.0333),
    "Baddi": (30.9578, 76.7914),
    "Nalagarh": (31.0410, 76.7200),
    "Paonta Sahib": (30.4366, 77.6248),
    "Nahan": (30.5595, 77.2955),
    "Muskura": (31.3000, 76.8000)
}


# ============================================================
# SEASON FUNCTION
# ============================================================

def get_season(month):
    """
    Returns the season used during model training.
    """

    if month in [12, 1, 2]:
        return "winter"

    elif month in [3, 4, 5]:
        return "summer"

    elif month in [6, 7, 8, 9]:
        return "monsoon"

    elif month in [10, 11]:
        return "post_monsoon"

    return "winter"


# ============================================================
# FESTIVAL DISTANCE FUNCTION
# ============================================================

def calculate_days_to_festival(selected_date):
    """
    Finds the nearest festival date and returns the signed
    number of days between the selected date and that festival.

    Positive value:
        Festival is upcoming.

    Negative value:
        Festival has recently passed.

    Zero:
        Festival is on the selected date.
    """

    if festival_df.empty:
        return 0

    festival_dates = festival_df["date"].dropna()

    if len(festival_dates) == 0:
        return 0

    differences = (
        festival_dates - selected_date
    ).dt.days

    nearest_index = differences.abs().idxmin()

    return int(differences.loc[nearest_index])


# ============================================================
# WEATHER FROM HISTORICAL DATASET
# ============================================================

def get_weather_from_dataset(market, selected_date):
    """
    First tries to get rainfall and temperature directly from
    the historical Kaggle dataset.

    This is preferred because the model was trained using these
    historical weather values.
    """

    if historical_df.empty:
        return None

    try:
        data = historical_df[
            (historical_df["market_name"].astype(str).str.strip() == market.strip())
            &
            (historical_df["date"].dt.date == selected_date.date())
        ]

        if data.empty:
            return None

        rainfall = data["rainfall_mm"].mean()
        temperature = data["temp_avg_c"].mean()

        if pd.isna(rainfall):
            rainfall = 0.0

        if pd.isna(temperature):
            temperature = 0.0

        return float(temperature), float(rainfall)

    except Exception as e:
        print("Error getting weather from dataset:", e)
        return None


# ============================================================
# WEATHER FROM OPEN-METEO API
# ============================================================

def get_weather_from_api(latitude, longitude, selected_date):
    """
    Uses Open-Meteo historical weather API as a fallback if
    weather data is not available in the historical dataset.
    """

    try:

        date_string = selected_date.strftime("%Y-%m-%d")

        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            f"&start_date={date_string}"
            f"&end_date={date_string}"
            "&daily=temperature_2m_mean,precipitation_sum"
            "&timezone=auto"
        )

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        weather_data = response.json()

        daily = weather_data.get("daily", {})

        temperatures = daily.get(
            "temperature_2m_mean",
            []
        )

        rainfall_values = daily.get(
            "precipitation_sum",
            []
        )

        if not temperatures:
            return None

        temperature = temperatures[0]

        if rainfall_values:
            rainfall = rainfall_values[0]
        else:
            rainfall = 0.0

        if temperature is None:
            temperature = 0.0

        if rainfall is None:
            rainfall = 0.0

        return float(temperature), float(rainfall)

    except Exception as e:
        print("Open-Meteo error:", e)
        return None


# ============================================================
# WEATHER FUNCTION
# ============================================================

def get_weather(market, selected_date):
    """
    Gets weather data.

    Priority:
    1. Historical Kaggle dataset
    2. Open-Meteo historical API
    """

    print(
        f"Getting weather for {market} on "
        f"{selected_date.strftime('%Y-%m-%d')}"
    )

    # --------------------------------------------------------
    # First try historical dataset
    # --------------------------------------------------------

    dataset_weather = get_weather_from_dataset(
        market,
        selected_date
    )

    if dataset_weather is not None:

        temperature, rainfall = dataset_weather

        print("Weather found in historical dataset.")
        print("Temperature:", temperature)
        print("Rainfall:", rainfall)

        return temperature, rainfall

    # --------------------------------------------------------
    # If dataset doesn't have weather, use API
    # --------------------------------------------------------

    if market not in market_coordinates:
        print("Market coordinates not available.")
        return None

    latitude, longitude = market_coordinates[market]

    print("Using Open-Meteo historical weather API.")

    api_weather = get_weather_from_api(
        latitude,
        longitude,
        selected_date
    )

    if api_weather is not None:

        temperature, rainfall = api_weather

        print("Temperature:", temperature)
        print("Rainfall:", rainfall)

        return temperature, rainfall

    return None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    # Get market names from the model dataset if possible
    markets = []

    if not historical_df.empty:
        try:
            markets = sorted(
                historical_df["market_name"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
        except Exception:
            markets = []

    # If dataset loading failed, use coordinate list
    if not markets:
        markets = sorted(
            market_coordinates.keys()
        )

    return render_template(
        "index.html",
        markets=markets
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None:
        return "Model could not be loaded.", 500

    # --------------------------------------------------------
    # Get form values
    # --------------------------------------------------------

    market = request.form.get(
        "market",
        ""
    ).strip()

    date_string = request.form.get(
        "date",
        ""
    ).strip()

    # --------------------------------------------------------
    # Validate market
    # --------------------------------------------------------

    if not market:
        return "Please select a market.", 400

    # --------------------------------------------------------
    # Validate date
    # --------------------------------------------------------

    try:

        selected_date = pd.to_datetime(
            date_string
        )

    except Exception:

        return "Invalid date.", 400

    # --------------------------------------------------------
    # Restrict prediction dates to model dataset period
    # --------------------------------------------------------

    minimum_date = pd.Timestamp(
        "2023-01-01"
    )

    maximum_date = pd.Timestamp(
        "2025-12-31"
    )

    if selected_date < minimum_date or selected_date > maximum_date:

        return (
            "Please select a date between "
            "January 1, 2023 and December 31, 2025."
        ), 400

    # --------------------------------------------------------
    # Get weather
    # --------------------------------------------------------

    weather = get_weather(
        market,
        selected_date
    )

    if weather is None:

        return (
            "Weather data could not be found for the "
            "selected market and date."
        ), 500

    temperature, rainfall = weather

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    days_to_festival = calculate_days_to_festival(
        selected_date
    )

    day_of_year = selected_date.dayofyear

    year = selected_date.year

    season = get_season(
        selected_date.month
    )

    # --------------------------------------------------------
    # Model input
    #
    # These MUST remain the same as the features used during
    # model training.
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "days_to_festival": [days_to_festival],
        "day_of_year": [day_of_year],
        "rainfall_mm": [rainfall],
        "temp_avg_c": [temperature],
        "market_name": [market],
        "season": [season],
        "year": [year]
    })

    print("Model input:")
    print(input_data)

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]

    except Exception as e:

        print("Prediction error:", e)

        return (
            "An error occurred while making the prediction."
        ), 500

    print(
        "Prediction in Rs./Quintal:",
        prediction
    )

    # ========================================================
    # CONVERT QUINTAL PRICE TO KG PRICE
    # ========================================================
    #
    # 1 quintal = 100 kg
    #
    # Model output:
    #       Rs./quintal
    #
    # Converted output:
    #       Rs./kg
    #
    # ========================================================

    prediction_per_kg = prediction / 100

    print(
        "Prediction in Rs./kg:",
        prediction_per_kg
    )

    # --------------------------------------------------------
    # Render result page
    # --------------------------------------------------------

    return render_template(
        "result.html",
        market=market,
        date=selected_date.strftime("%Y-%m-%d"),

        # Display converted price
        prediction=round(
            prediction_per_kg,
            2
        ),

        days_to_festival=days_to_festival,

        rainfall=round(
            rainfall,
            2
        ),

        temperature=round(
            temperature,
            2
        )
    )


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history():

    return render_template(
        "history.html"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )