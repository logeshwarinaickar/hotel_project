import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import date

@st.cache_data
def create_daily_occupancy(df):
    
    df = df.copy()

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_date_month"] + "-" +
        df["arrival_date_day_of_month"].astype(str)
    )

    df["total_stay_nights"] = (
        df["stays_in_weekend_nights"] +
        df["stays_in_week_nights"]
    )

    df_stays = df[
        (df["is_canceled"] == 0) &
        (df["total_stay_nights"] > 0)
    ].copy()

    df_stays["checkout_date"] = (
        df_stays["arrival_date"] +
        pd.to_timedelta(
            df_stays["total_stay_nights"],
            unit="D"
        )
    )

    occupied_records = []

    for _, row in df_stays.iterrows():

        stay_dates = pd.date_range(
            start=row["arrival_date"],
            end=row["checkout_date"] - pd.Timedelta(days=1),
            freq="D"
        )

        for stay_date in stay_dates:
            occupied_records.append(stay_date)

    occupied_df = pd.DataFrame({
        "date": occupied_records
    })

    daily_occupancy = (
        occupied_df
        .groupby("date")
        .size()
        .reset_index(name="occupied_rooms")
    )

    daily_occupancy = (
        daily_occupancy
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily_occupancy

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hotel Booking & Occupancy Dashboard",
    page_icon="🏨",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM UI STYLING
# ------------------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #eaf6ff;
    }

    /* Main content width and spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.2rem !important;
    }

    /* Section headings */
   h2 {
        font-size: 1.9rem !important;
        font-weight: 650 !important;
        margin-top: 1.5rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
    }

    /* Normal text */
    p {
        line-height: 1.6;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Input boxes */
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div,
    .stDateInput > div > div {
        border-radius: 8px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 12px;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* White input fields */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div {
        background-color: white !important;
        border-radius: 8px !important;
    }

    /* White date input */
    div[data-testid="stDateInput"] input {
        background-color: white !important;
    }

    /* White number inputs */
    div[data-testid="stNumberInput"] input {
        background-color: white !important;
    }

    /* White selectbox */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: white !important;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================

# Final Machine Learning model
rf_model = joblib.load("hotel_cancellation_risk_model.pkl")

# Final Deep Learning model
lstm_model = load_model("hotel_occupancy_lstm_model.keras")

# Occupancy scaler
scaler = joblib.load("hotel_occupancy_scaler.pkl")

# Load original hotel booking dataset
df_dl = pd.read_csv("hotel_bookings.csv")



# ============================================================
# TITLE
# ============================================================

st.title("🏨 Hotel Booking & Occupancy Dashboard")

st.write(
    "AI-powered hotel booking cancellation risk and occupancy "
    "forecasting system."
)

st.divider()
# ============================================================
# DAILY OCCUPANCY DATA FOR LSTM
# ============================================================

daily_occupancy = create_daily_occupancy(df_dl)




# ============================================================
# CANCELLATION RISK
# ============================================================

st.header("🎯 Booking Cancellation Risk")

st.write(
    "Enter the booking details available at the time of reservation "
    "to estimate the cancellation risk."
)

# ============================================================
# BOOKING DETAILS
# ============================================================

st.subheader("📋 Booking Details")

col1, col2 = st.columns(2)

# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    hotel = st.selectbox(
        " Hotel Type",
        ["City Hotel", "Resort Hotel"]
    )

    arrival_date = st.date_input(
        " Expected Arrival Date",
        value=date.today()
    )

    lead_time = st.number_input(
        " How many days before arrival was the booking made?",
        min_value=0,
        value=30,
        help=(
            "Example: If the guest booked 30 days before "
            "their arrival, enter 30."
        )
    )

    stays_in_weekend_nights = st.number_input(
        " Weekend Nights",
        min_value=0,
        value=1
    )

    stays_in_week_nights = st.number_input(
        " Weekday Nights",
        min_value=0,
        value=2
    )

    adults = st.number_input(
        " Adults",
        min_value=1,
        value=2
    )

    children = st.number_input(
        " Children",
        min_value=0,
        value=0
    )

# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    babies = st.number_input(
        " Babies",
        min_value=0,
        value=0
    )

    is_repeated_guest = st.selectbox(
       " Has this guest stayed at the hotel before?",
       [0, 1],
       format_func=lambda x: "Yes" if x == 1 else "No"
    )

    if is_repeated_guest == 1:

      previous_cancellations = st.number_input(
        " How many of the guest's previous bookings were cancelled?",
        min_value=0,
        value=0,
        help="Enter the number of previous hotel bookings cancelled by this guest."
    )

    else:

       previous_cancellations = 0

    adr = st.number_input(
        " Average Daily Rate (ADR)",
        min_value=0.0,
        value=100.0
    )

    required_car_parking_spaces = st.number_input(
        " Parking Spaces Required",
        min_value=0,
        value=0
    )

    total_of_special_requests = st.number_input(
        " Special Requests",
        min_value=0,
        value=0
    )

# ============================================================
# AUTOMATICALLY CALCULATED FEATURES
# ============================================================

# Total length of stay
total_stay_nights = (
    stays_in_weekend_nights +
    stays_in_week_nights
)

# Total number of guests
total_guests = (
    adults +
    children +
    babies
)

# Extract date-related features automatically
arrival_date_year = arrival_date.year
arrival_month_num = arrival_date.month
arrival_day_of_week = arrival_date.weekday()
arrival_week = arrival_date.isocalendar().week

# This feature is part of the trained model but is not
# directly entered by the hotel manager.
previous_bookings_not_canceled = 0

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Cancellation Risk",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = {

        # Categorical features
        "hotel": [hotel],
        "meal": ["BB"],
        "country": ["PRT"],
        "market_segment": ["Online TA"],
        "distribution_channel": ["TA/TO"],
        "reserved_room_type": ["A"],
        "deposit_type": ["No Deposit"],
        "customer_type": ["Transient"],

        # Numerical features
        "lead_time": [lead_time],
        "arrival_date_year": [arrival_date_year],
        "stays_in_weekend_nights": [
            stays_in_weekend_nights
        ],
        "stays_in_week_nights": [
            stays_in_week_nights
        ],
        "adults": [adults],
        "children": [children],
        "babies": [babies],
        "is_repeated_guest": [is_repeated_guest],
        "previous_cancellations": [
            previous_cancellations
        ],
        "previous_bookings_not_canceled": [
            previous_bookings_not_canceled
        ],
        "adr": [adr],
        "required_car_parking_spaces": [
            required_car_parking_spaces
        ],
        "total_of_special_requests": [
            total_of_special_requests
        ],
        "total_stay_nights": [
            total_stay_nights
        ],
        "total_guests": [total_guests],
        "arrival_month_num": [
            arrival_month_num
        ],
        "arrival_day_of_week": [
            arrival_day_of_week
        ],
        "arrival_week": [arrival_week]
    }

    input_df = pd.DataFrame(input_data)

    # --------------------------------------------------------
    # Predict cancellation probability
    # --------------------------------------------------------

    probability = rf_model.predict_proba(
        input_df
    )[0][1]

    probability_percent = probability * 100

    # --------------------------------------------------------
    # Determine risk level
    # --------------------------------------------------------

    if probability < 0.30:

        risk = "LOW"

    elif probability < 0.60:

        risk = "MEDIUM"

    else:

        risk = "HIGH"

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Cancellation Risk Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Cancellation Probability",
            f"{probability_percent:.1f}%"
        )

    with result_col2:

        if risk == "LOW":

            st.success("🟢 LOW RISK")

        elif risk == "MEDIUM":

            st.warning("🟡 MEDIUM RISK")

        else:

            st.error("🔴 HIGH RISK")

    st.info(
        f"Based on the booking information provided, "
        f"the estimated cancellation probability is "
        f"**{probability_percent:.1f}%**."
    )

# ============================================================
# LSTM OCCUPANCY FORECAST
# ============================================================

st.divider()

st.header("📈 Future Occupancy Forecast")

st.write(
    "Use historical occupancy patterns learned by the LSTM model "
    "to generate a 7-day occupancy forecast."
)

# ------------------------------------------------------------
# Select forecast start date
# ------------------------------------------------------------

forecast_start_date = st.date_input(
    "📅 Select Forecast Start Date",
    value=date.today()
)

# ------------------------------------------------------------
# Generate 7-day forecast
# ------------------------------------------------------------

if len(daily_occupancy) >= 7:

    # --------------------------------------------------------
    # Find historical data corresponding to selected date
    # --------------------------------------------------------

    selected_date = pd.Timestamp(forecast_start_date)

    selected_month = selected_date.month
    selected_day = selected_date.day

    # Use the most recent historical year available
    historical_data = daily_occupancy[
        (daily_occupancy["date"].dt.month == selected_month) &
        (daily_occupancy["date"].dt.day == selected_day)
    ].copy()

    # --------------------------------------------------------
    # Select the closest historical matching date
    # --------------------------------------------------------

    if len(historical_data) > 0:

        historical_start_date = historical_data["date"].max()

        # Get the 7 days BEFORE the matching historical date
        historical_sequence = daily_occupancy[
            (daily_occupancy["date"] < historical_start_date)
        ].tail(7)

    else:

        # Fallback if exact month/day is not available
        historical_sequence = daily_occupancy.tail(7)

    # --------------------------------------------------------
    # Make sure we have 7 historical values
    # --------------------------------------------------------

    if len(historical_sequence) == 7:

        last_7_values = (
            historical_sequence["occupied_rooms"]
            .values
            .reshape(-1, 1)
        )

        # Scale using the SAME scaler used during training
        current_sequence = scaler.transform(
            last_7_values
        ).flatten()

        predictions = []

        # ----------------------------------------------------
        # Generate 7 predictions recursively
        # ----------------------------------------------------

        for i in range(7):

            X_input = current_sequence.reshape(
                1, 7, 1
            )

            predicted_scaled = lstm_model.predict(
                X_input,
                verbose=0
            )[0][0]

            predicted_rooms = scaler.inverse_transform(
                [[predicted_scaled]]
            )[0][0]

            predicted_rooms = max(
                0,
                round(predicted_rooms)
            )

            predictions.append(predicted_rooms)

            # Remove oldest value and add new prediction
            current_sequence = np.append(
                current_sequence[1:],
                predicted_scaled
            )

        # ----------------------------------------------------
        # Create forecast dates
        # ----------------------------------------------------

        forecast_dates = pd.date_range(
            start=pd.Timestamp(forecast_start_date),
            periods=7,
            freq="D"
        )

        # ----------------------------------------------------
        # Forecast table
        # ----------------------------------------------------

        forecast_df = pd.DataFrame({
            "Date": forecast_dates.strftime("%d %B %Y"),
            "Expected Occupied Rooms": predictions
        })

        st.subheader("🔮 7-Day Occupancy Forecast")

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Forecast chart
        # ----------------------------------------------------

        st.subheader("📊 Forecast Trend")

        chart_df = pd.DataFrame({
            "Date": forecast_dates,
            "Expected Occupied Rooms": predictions
        })

        st.line_chart(
            chart_df.set_index("Date")
        )

    else:

        st.warning(
            "Not enough historical data available "
            "for this forecast period."
        )

else:

    st.warning(
        "Not enough historical occupancy data available "
        "to generate a forecast."
    )