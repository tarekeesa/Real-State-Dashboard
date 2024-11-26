# datapipeline/feature_engineering.py

import pandas as pd
import logging
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False)
def perform_feature_engineering(rent, transactions):
    logger.info("Starting feature engineering")
    
    # Define the columns to be used for PROPERTY_ID
    rent_columns = ['AREA_EN', 'PROP_TYPE_EN', 'PROP_SUB_TYPE_EN', 'PROJECT_EN']
    transactions_columns = ['AREA_EN', 'PROP_TYPE_EN', 'PROP_SB_TYPE_EN', 'PROJECT_EN']
    
    # Handle missing values by filling with 'Unknown'
    # First, add 'Unknown' to Categorical columns if they are categorical
    for col in rent_columns:
        if col in rent.columns:
            if pd.api.types.is_categorical_dtype(rent[col]):
                if 'Unknown' not in rent[col].cat.categories:
                    rent[col] = rent[col].cat.add_categories(['Unknown'])
                    logger.debug(f"Added 'Unknown' category to '{col}' in rent DataFrame")
        else:
            logger.warning(f"Column '{col}' not found in rent DataFrame")
    
    for col in transactions_columns:
        if col in transactions.columns:
            if pd.api.types.is_categorical_dtype(transactions[col]):
                if 'Unknown' not in transactions[col].cat.categories:
                    transactions[col] = transactions[col].cat.add_categories(['Unknown'])
                    logger.debug(f"Added 'Unknown' category to '{col}' in transactions DataFrame")
        else:
            logger.warning(f"Column '{col}' not found in transactions DataFrame")
    
    # Now, fill NaN with 'Unknown' for the specified columns
    rent = rent.fillna({'AREA_EN': 'Unknown',
                        'PROP_TYPE_EN': 'Unknown',
                        'PROP_SUB_TYPE_EN': 'Unknown',
                        'PROJECT_EN': 'Unknown'})
    logger.info("Filled NaN values with 'Unknown' in rent DataFrame")
    
    transactions = transactions.fillna({'AREA_EN': 'Unknown',
                                        'PROP_TYPE_EN': 'Unknown',
                                        'PROP_SB_TYPE_EN': 'Unknown',
                                        'PROJECT_EN': 'Unknown'})
    logger.info("Filled NaN values with 'Unknown' in transactions DataFrame")
    
    # Convert Categorical columns to string if necessary
    # This step is optional and depends on whether you want to keep them as Categorical or not
    # If you decide to keep them as Categorical, you can skip this step
    # Below is an example of converting them to string
    for col in rent_columns:
        if col in rent.columns:
            rent[col] = rent[col].astype(str)
            logger.debug(f"Converted '{col}' to string in rent DataFrame")
    
    for col in transactions_columns:
        if col in transactions.columns:
            transactions[col] = transactions[col].astype(str)
            logger.debug(f"Converted '{col}' to string in transactions DataFrame")
    
    # Create unique PROPERTY_ID for rent DataFrame
    if all(col in rent.columns for col in rent_columns):
        rent['PROPERTY_ID'] = (
            rent['AREA_EN'] + '_' +
            rent['PROP_TYPE_EN'] + '_' +
            rent['PROP_SUB_TYPE_EN'] + '_' +
            rent['PROJECT_EN']
        )
        logger.info("Created 'PROPERTY_ID' in rent DataFrame")
    else:
        missing_cols = [col for col in rent_columns if col not in rent.columns]
        logger.error(f"Cannot create 'PROPERTY_ID' for rent DataFrame. Missing columns: {missing_cols}")
        st.error(f"Cannot create 'PROPERTY_ID' for rent data. Missing columns: {', '.join(missing_cols)}")
        rent['PROPERTY_ID'] = pd.NA  # Assign NaN if columns are missing
    
    # Create unique PROPERTY_ID for transactions DataFrame
    if all(col in transactions.columns for col in transactions_columns):
        transactions['PROPERTY_ID'] = (
            transactions['AREA_EN'] + '_' +
            transactions['PROP_TYPE_EN'] + '_' +
            transactions['PROP_SB_TYPE_EN'] + '_' +
            transactions['PROJECT_EN']
        )
        logger.info("Created 'PROPERTY_ID' in transactions DataFrame")
    else:
        missing_cols = [col for col in transactions_columns if col not in transactions.columns]
        logger.error(f"Cannot create 'PROPERTY_ID' for transactions DataFrame. Missing columns: {missing_cols}")
        st.error(f"Cannot create 'PROPERTY_ID' for transactions data. Missing columns: {', '.join(missing_cols)}")
        transactions['PROPERTY_ID'] = pd.NA  # Assign NaN if columns are missing
    
    logger.info("Completed feature engineering")
    return rent, transactions
