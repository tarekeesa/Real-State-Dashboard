# datapipeline/cleaning.py

import pandas as pd
import logging
from datapipeline.utils import standardize_categorical_columns, validate_data, count_outliers
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_rent_df(rent_df):
    logger.info("Starting cleaning of rent DataFrame")

    # Validate 'ACTUAL_AREA' and 'ANNUAL_AMOUNT'
    rent_df = validate_data(rent_df, ['ACTUAL_AREA', 'ANNUAL_AMOUNT'])

    # Convert date columns to datetime
    date_cols = ['REGISTRATION_DATE', 'START_DATE', 'END_DATE']
    for col in date_cols:
        rent_df[col] = pd.to_datetime(rent_df[col], errors='coerce')
        logger.debug(f"Converted column {col} to datetime")

    # Convert numerical columns to appropriate types
    num_cols = ['CONTRACT_AMOUNT', 'ANNUAL_AMOUNT', 'ACTUAL_AREA', 'ROOMS', 'PARKING', 'TOTAL_PROPERTIES']
    for col in num_cols:
        rent_df[col] = pd.to_numeric(rent_df[col], errors='coerce')
        logger.debug(f"Converted column {col} to numeric")

    # Drop 'MASTER_PROJECT_EN' due to high missingness
    rent_df = rent_df.drop(columns=['MASTER_PROJECT_EN'], errors='ignore')
    logger.info("Dropped column 'MASTER_PROJECT_EN'")

    # Handle 'NEAREST_METRO_EN', 'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN'
    for col in ['NEAREST_METRO_EN', 'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN']:
        rent_df[col] = rent_df[col].fillna('Unknown')
        logger.debug(f"Filled missing values in column {col} with 'Unknown'")

    # Handle 'ROOMS'
    rent_df['ROOMS'] = rent_df['ROOMS'].fillna('Not_Applicable').astype(str)
    logger.info("Handled missing values in 'ROOMS' column")

    # Convert 'PARKING' to string before assignments
    rent_df['PARKING'] = rent_df['PARKING'].astype(str)
    logger.debug("Converted 'PARKING' column to string")

    # Handle 'PARKING' conditionally based on property type
    villa_property_type = 'Villa'
    parking_for_villa = 'Available'
    parking_for_others = 'No_Parking'

    # Impute 'PARKING' for Villas
    rent_df.loc[
        (rent_df['PARKING'].isnull()) &
        (rent_df['PROP_TYPE_EN'].str.strip().str.lower() == villa_property_type.lower()),
        'PARKING'
    ] = parking_for_villa

    # Impute 'PARKING' for Non-Villas
    rent_df.loc[
        (rent_df['PARKING'].isnull()) &
        (rent_df['PROP_TYPE_EN'].str.strip().str.lower() != villa_property_type.lower()),
        'PARKING'
    ] = parking_for_others

    logger.info("Imputed 'PARKING' column based on property type")

    # Handle 'PROP_SUB_TYPE_EN'
    rent_df['PROP_SUB_TYPE_EN'] = rent_df['PROP_SUB_TYPE_EN'].fillna('Not_Known')
    logger.debug("Filled missing values in 'PROP_SUB_TYPE_EN' with 'Not_Known'")

    # Handle 'USAGE_EN'
    rent_df['USAGE_EN'] = rent_df['USAGE_EN'].fillna('Unknown')
    logger.debug("Filled missing values in 'USAGE_EN' with 'Unknown'")

    # Assign 'Unknown' to missing 'PROJECT_EN' values
    rent_df['PROJECT_EN'] = rent_df['PROJECT_EN'].fillna('Unknown')
    logger.debug("Filled missing values in 'PROJECT_EN' with 'Unknown'")

    # Drop rows with missing or invalid critical values
    initial_row_count = rent_df.shape[0]
    rent_df = rent_df.dropna(subset=['ANNUAL_AMOUNT', 'ACTUAL_AREA'])
    logger.info(f"Dropped {initial_row_count - rent_df.shape[0]} rows with missing 'ANNUAL_AMOUNT' or 'ACTUAL_AREA'")

    # Exclude invalid 'ACTUAL_AREA' and 'ANNUAL_AMOUNT'
    rent_df = rent_df[rent_df['ACTUAL_AREA'] > 0]
    rent_df = rent_df[rent_df['ANNUAL_AMOUNT'] > 0]
    logger.info("Excluded entries with non-positive 'ACTUAL_AREA' or 'ANNUAL_AMOUNT'")

    # Count outliers
    numerical_columns = ['ANNUAL_AMOUNT', 'ACTUAL_AREA']
    rent_outliers = count_outliers(rent_df, numerical_columns)
    logger.info(f"Identified outliers in rent DataFrame: {rent_outliers}")

    logger.info("Completed cleaning of rent DataFrame")
    return rent_df, rent_outliers

def clean_transactions_df(transactions_df):
    logger.info("Starting cleaning of transactions DataFrame")

    # Validate 'ACTUAL_AREA' and 'TRANS_VALUE'
    transactions_df = validate_data(transactions_df, ['ACTUAL_AREA', 'TRANS_VALUE'])

    # Convert date columns to datetime
    transactions_df['INSTANCE_DATE'] = pd.to_datetime(transactions_df['INSTANCE_DATE'], errors='coerce')
    logger.debug("Converted 'INSTANCE_DATE' to datetime")

    # Convert numerical columns to appropriate types
    num_cols = ['TRANS_VALUE', 'ACTUAL_AREA', 'TOTAL_BUYER', 'TOTAL_SELLER']
    for col in num_cols:
        transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce')
        logger.debug(f"Converted column {col} to numeric")

    # Drop 'MASTER_PROJECT_EN' due to high missingness
    transactions_df = transactions_df.drop(columns=['MASTER_PROJECT_EN'], errors='ignore')
    logger.info("Dropped column 'MASTER_PROJECT_EN'")

    # Handle 'NEAREST_METRO_EN', 'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN'
    for col in ['NEAREST_METRO_EN', 'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN']:
        transactions_df[col] = transactions_df[col].fillna('Unknown')
        logger.debug(f"Filled missing values in column {col} with 'Unknown'")

    # Handle 'ROOMS_EN'
    transactions_df['ROOMS_EN'] = transactions_df['ROOMS_EN'].fillna('Not_Applicable').astype(str)
    logger.info("Handled missing values in 'ROOMS_EN' column")

    # Convert 'PARKING' to string before assignments
    transactions_df['PARKING'] = transactions_df['PARKING'].astype(str)
    logger.debug("Converted 'PARKING' column to string")

    # Handle 'PARKING' conditionally based on property type
    villa_property_type = 'Villa'
    parking_for_villa = 'Available'
    parking_for_others = 'No_Parking'

    # Impute 'PARKING' for Villas
    transactions_df.loc[
        (transactions_df['PARKING'].isnull()) &
        (transactions_df['PROP_TYPE_EN'].str.strip().str.lower() == villa_property_type.lower()),
        'PARKING'
    ] = parking_for_villa

    # Impute 'PARKING' for Non-Villas
    transactions_df.loc[
        (transactions_df['PARKING'].isnull()) &
        (transactions_df['PROP_TYPE_EN'].str.strip().str.lower() != villa_property_type.lower()),
        'PARKING'
    ] = parking_for_others

    logger.info("Imputed 'PARKING' column based on property type")

    # Handle 'PROP_SB_TYPE_EN'
    transactions_df['PROP_SB_TYPE_EN'] = transactions_df['PROP_SB_TYPE_EN'].fillna('Not_Known')
    logger.debug("Filled missing values in 'PROP_SB_TYPE_EN' with 'Not_Known'")

    # Handle 'PROCEDURE_AREA'
    transactions_df['PROCEDURE_AREA'] = pd.to_numeric(transactions_df['PROCEDURE_AREA'], errors='coerce')
    transactions_df['PROCEDURE_AREA'] = transactions_df['PROCEDURE_AREA'].fillna(transactions_df['PROCEDURE_AREA'].median())
    logger.info("Handled 'PROCEDURE_AREA' missing values by filling with median")

    # Assign 'Unknown' to missing 'PROJECT_EN' values
    transactions_df['PROJECT_EN'] = transactions_df['PROJECT_EN'].fillna('Unknown')
    logger.debug("Filled missing values in 'PROJECT_EN' with 'Unknown'")

    # Drop rows with missing or invalid critical values
    initial_row_count = transactions_df.shape[0]
    transactions_df = transactions_df.dropna(subset=['TRANS_VALUE', 'ACTUAL_AREA'])
    logger.info(f"Dropped {initial_row_count - transactions_df.shape[0]} rows with missing 'TRANS_VALUE' or 'ACTUAL_AREA'")

    # Exclude invalid 'ACTUAL_AREA' and 'TRANS_VALUE'
    transactions_df = transactions_df[transactions_df['ACTUAL_AREA'] > 0]
    transactions_df = transactions_df[transactions_df['TRANS_VALUE'] > 0]
    logger.info("Excluded entries with non-positive 'ACTUAL_AREA' or 'TRANS_VALUE'")

    # Count outliers
    numerical_columns = ['TRANS_VALUE', 'ACTUAL_AREA']
    trans_outliers = count_outliers(transactions_df, numerical_columns)
    logger.info(f"Identified outliers in transactions DataFrame: {trans_outliers}")

    logger.info("Completed cleaning of transactions DataFrame")
    return transactions_df, trans_outliers

def impute_missing_values(df, strategy='median'):
    logger.info(f"Imputing missing values using strategy: {strategy}")
    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype in ['float64', 'int64']:
                if strategy == 'median':
                    df[column] = df[column].fillna(df[column].median())
                    logger.debug(f"Filled missing values in numeric column '{column}' with median")
                elif strategy == 'mean':
                    df[column] = df[column].fillna(df[column].mean())
                    logger.debug(f"Filled missing values in numeric column '{column}' with mean")
                elif strategy == 'zero':
                    df[column] = df[column].fillna(0)
                    logger.debug(f"Filled missing values in numeric column '{column}' with zero")
                else:
                    logger.warning(f"Unknown imputation strategy '{strategy}' for column '{column}'")
            else:
                df[column] = df[column].fillna('Unknown')
                logger.debug(f"Filled missing values in non-numeric column '{column}' with 'Unknown'")
    logger.info("Completed missing value imputation")
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def clean_and_detect_outliers(transactions, rent):
    logger.info("Starting data cleaning and outlier detection")

    # Standardize categorical columns
    categorical_columns = ['AREA_EN', 'PROP_TYPE_EN', 'PROP_SB_TYPE_EN', 'USAGE_EN', 'ROOMS_EN', 'PARKING']
    transactions = standardize_categorical_columns(transactions, categorical_columns)
    rent = standardize_categorical_columns(rent, categorical_columns)

    # Clean DataFrames and get outlier counts
    rent_cleaned, rent_outliers = clean_rent_df(rent)
    transactions_cleaned, trans_outliers = clean_transactions_df(transactions)

    logger.info("Completed data cleaning and outlier detection")
    return transactions_cleaned, rent_cleaned, trans_outliers, rent_outliers
