# 📌 AI-BASED HOTEL BOOKING & OCCUPANCY FORECASTING SYSTEM

## 🚀 Project Overview

This is a Machine Learning and Deep Learning web application built using Streamlit that predicts hotel booking cancellation risk and forecasts future hotel occupancy.

The system combines a **Tuned Random Forest model** for booking cancellation prediction and an **LSTM model** for 7-day occupancy forecasting.

---

## 🧠 Problem Statement

Hotel cancellations can affect revenue, room availability, and operational planning. At the same time, forecasting future occupancy can help hotels prepare for expected demand.

The goal of this project is:

* Predict whether a hotel booking is likely to be cancelled
* Estimate the cancellation probability and risk level
* Forecast hotel occupancy for the next 7 days
* Provide an interactive dashboard for hotel analytics

---

## ⚙️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest (ML Model)
* TensorFlow / Keras
* LSTM (Deep Learning Model)
* Streamlit
* Joblib
* Matplotlib
* Git & GitHub

---

## 📊 Dataset

The project uses the **Hotel Booking Demand Dataset** containing hotel reservation information.

* Original records: 119,390
* Records after preprocessing: 87,229
* Hotels: City Hotel & Resort Hotel
* Historical data: 2015–2017

The dataset contains information such as booking details, guest information, stay duration, arrival dates, previous booking history, ADR, special requests, and cancellation status.

---

## 🔍 Features Used

The cancellation prediction model uses booking information available at the time of reservation, including:

* Hotel Type
* Lead Time
* Arrival Date
* Weekend Nights
* Weekday Nights
* Number of Adults
* Number of Children
* Number of Babies
* Previous Cancellations
* ADR
* Required Parking Spaces
* Special Requests
* Guest history
* Booking and customer-related information

Leakage-prone fields such as `reservation_status` and `reservation_status_date` were excluded from the prediction model.

---

## 🖥️ App Features

* Streamlit-based interactive dashboard
* Real-time booking cancellation prediction
* Cancellation probability score
* Cancellation risk classification:

  * 🟢 Low Risk
  * 🟡 Medium Risk
  * 🔴 High Risk
* 7-day future occupancy forecasting
* Interactive occupancy forecast chart
* Booking and occupancy analysis in one dashboard

---

## 🤖 Machine Learning Model

**Model Used:** Tuned Random Forest Classifier

The model uses a time-based train-test split, with earlier years used for training and 2017 used as the test period.

---

## 🧠 Deep Learning Model

**Model Used:** LSTM (Long Short-Term Memory)

The LSTM model learns historical daily occupancy patterns and generates a **7-day occupancy forecast**.

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/logeshwarinaickar/hotel_project.git
```

Go to the project folder:

```bash
cd hotel_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
hotel_project/
├── app.py
├── hotel_bookings.csv
├── hotel_cancellation_risk_model.pkl
├── hotel_occupancy_lstm_model.keras
├── hotel_occupancy_scaler.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🌐 Live Demo

👉 **Streamlit App:**
https://hotelproject-kw26e2mnqcnaq7in9nexen.streamlit.app/

---

## 📈 Project Highlights

* Combines **Machine Learning + Deep Learning**
* Uses a **Tuned Random Forest** for cancellation risk prediction
* Uses **LSTM** for occupancy forecasting
* Includes data preprocessing and feature engineering
* Uses leakage-aware feature selection
* Provides an interactive Streamlit dashboard
* Uses historical occupancy patterns to generate 7-day forecasts

---

## ⚠️ Limitations

The occupancy forecasting model is trained on historical data from approximately 2015–2017 and reflects the occupancy patterns present in that dataset.

Therefore, the forecast should be interpreted as a model-based prediction rather than a direct prediction of the actual capacity of a specific hotel.

---

## 🧑‍💻 Author

* **Logeshwari Naickar**
* B.Sc. Data Science
* Data Science & Machine Learning Enthusiast

---

## ⭐ Acknowledgements

* Hotel Booking Demand Dataset
* Scikit-learn documentation
* TensorFlow / Keras documentation
* Streamlit documentation
* Kaggle
