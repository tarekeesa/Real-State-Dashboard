# datapipeline/feature_engineering.py
import pandas as pd
import logging
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False)
def perform_feature_engineering(rent, transactions):
    logger.info("Starting feature engineering")

    # Fill missing values in concatenated columns
    columns_to_fill_rent = ['AREA_EN', 'PROP_TYPE_EN', 'PROP_SUB_TYPE_EN', 'PROJECT_EN']
    rent[columns_to_fill_rent] = rent[columns_to_fill_rent].fillna('Unknown')
    logger.debug("Filled missing values in rent DataFrame concatenated columns")

    columns_to_fill_trans = ['AREA_EN', 'PROP_TYPE_EN', 'PROP_SB_TYPE_EN', 'PROJECT_EN']
    transactions[columns_to_fill_trans] = transactions[columns_to_fill_trans].fillna('Unknown')
    logger.debug("Filled missing values in transactions DataFrame concatenated columns")

    # Create unique property identifiers
    rent['PROPERTY_ID'] = (
        rent['AREA_EN'] + '_' +
        rent['PROP_TYPE_EN'] + '_' +
        rent['PROP_SUB_TYPE_EN'] + '_' +
        rent['PROJECT_EN']
    )
    logger.info("Created 'PROPERTY_ID' in rent DataFrame")

    transactions['PROPERTY_ID'] = (
        transactions['AREA_EN'] + '_' +
        transactions['PROP_TYPE_EN'] + '_' +
        transactions['PROP_SB_TYPE_EN'] + '_' +
        transactions['PROJECT_EN']
    )
    logger.info("Created 'PROPERTY_ID' in transactions DataFrame")

    logger.info("Completed feature engineering")
    return rent, transactions
