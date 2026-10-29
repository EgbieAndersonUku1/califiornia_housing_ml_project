
import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"


st.title("🏠 California Housing Price Prediction")

st.markdown(
    """
    **An end-to-end machine learning application**

    Predicting median housing values using neural networks,
    MLflow, FastAPI, Docker, and Streamlit.
    """
)

st.divider()

st.header("🎯 Project Aim")

st.write(
    "To develop and deploy a machine learning system "
    "that predicts median house values in California "
    "using geographic, demographic, and housing data."
)

st.subheader("The Business Problem")

st.write(
    "Housing prices vary across California due to "
    "location, income, population, and housing characteristics. "
    "This project explores how neural networks can support "
    "data-driven housing market analysis."
)

st.subheader("Project Objectives")

st.markdown(
    """
    - Explore and preprocess the housing dataset.
    - Train and compare three neural networks.
    - Evaluate predictive accuracy.
    - Track experiments with MLflow.
    - Deploy the selected model using FastAPI and Docker.
    - Provide an interactive Streamlit interface.
    """
)

st.divider()


st.subheader("Enter Housing Information")

with st.form("housing_form"):
    longitude = st.number_input(
        "Longitude", value=-122.23
    )
    latitude = st.number_input(
        "Latitude", value=37.88
    )
    housing_median_age = st.number_input(
        "Housing Median Age", min_value=0.0, value=41.0
    )
    total_rooms = st.number_input(
        "Total Rooms", min_value=0.0, value=880.0
    )
    total_bedrooms = st.number_input(
        "Total Bedrooms", min_value=0.0, value=129.0
    )
    population = st.number_input(
        "Population", min_value=0.0, value=322.0
    )
    households = st.number_input(
        "Households", min_value=0.0, value=126.0
    )
    median_income = st.number_input(
        "Median Income (dataset units)",
        min_value=0.0,
        value=8.3252,
        help="The dataset represents median income in units of $10,000.",
    )
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        [
            "NEAR BAY",
            "INLAND",
            "<1H OCEAN",
            "NEAR OCEAN",
            "ISLAND",
        ],
    )

    submitted = st.form_submit_button(
        "Predict House Value",
        type="primary",
    )

if submitted:
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()

        st.success("Prediction completed successfully!")

        st.metric(
            "Estimated Median House Value",
            f"${result['predicted_house_value']:,.2f}",
        )

        st.caption(
            f"Prediction generated using {result['model']}."
        )

    except requests.RequestException as error:
        st.error(f"Prediction request failed: {error}")
