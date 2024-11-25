import pandas as pd
import streamlit as st
import logging

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False, ttl=3600)
def preprocess_source_data(rent, transactions):
    logger.info("Starting data preprocessing")

    # Define the analysis period (e.g., last year)
    analysis_start_date = pd.to_datetime('2023-10-01')
    analysis_end_date = pd.to_datetime('2024-09-30')

    # Filter data within the analysis period
    rent = rent[
        (rent['REGISTRATION_DATE'] >= analysis_start_date) & 
        (rent['REGISTRATION_DATE'] <= analysis_end_date)
    ]
    transactions = transactions[
        (transactions['INSTANCE_DATE'] >= analysis_start_date) & 
        (transactions['INSTANCE_DATE'] <= analysis_end_date)
    ]
    logger.info("Filtered data within the analysis period")

    # Extract month and year as strings
    rent['Month'] = rent['REGISTRATION_DATE'].dt.strftime('%Y-%m')
    transactions['Month'] = transactions['INSTANCE_DATE'].dt.strftime('%Y-%m')
    logger.info("Extracted 'Month' from date columns")

    logger.info("Completed data preprocessing")
    return rent, transactions
