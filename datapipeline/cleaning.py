# datapipeline/cleaning.py

import pandas as pd
import numpy as np
import logging
import streamlit as st
import requests

from datapipeline.data_converting import downcast_float, downcast_integer, optimize_transactions_dataframe
from datapipeline.utils import standardize_categorical_columns, validate_data, count_outliers, get_column_case_insensitive

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data(df, columns):
    """
    Validates that specified columns in the DataFrame have no zero or negative values.
    Drops rows where invalid values are found.
    """
    for col in columns:
        actual_col = get_column_case_insensitive(df, col)
        if actual_col:
            invalid_count = (df[actual_col] <= 0).sum()
            if invalid_count > 0:
                df = df[df[actual_col] > 0]
                logger.info(f"Excluded {invalid_count} rows with non-positive values in '{actual_col}'")
        else:
            st.warning(f"Data Validation Warning: Column '{col}' not found in DataFrame. Skipping validation for this column.")
            logger.warning(f"Column '{col}' not found in DataFrame.")
    return df

def impute_missing_numeric(df, strategy='median', numeric_columns=None):
    """
    Imputes missing values in numeric columns based on the specified strategy.
    """
    logger.info(f"Imputing missing values in numeric columns using strategy: {strategy}")
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['float64', 'float32', 'float16', 'int64', 'int32', 'uint64', 'uint32']).columns.tolist()
    for column in numeric_columns:
        if df[column].isnull().any():
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
    logger.info("Completed numeric missing value imputation")
    return df

def fill_categorical_na(df, categorical_columns, new_category='Unknown'):
    """
    Adds 'Unknown' category to specified categorical columns and fills missing values with 'Unknown'.
    """
    logger.info("Filling missing values in categorical columns with 'Unknown'")
    for column in categorical_columns:
        if column in df.columns:
            if pd.api.types.is_categorical_dtype(df[column]):
                if new_category not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([new_category])
                    logger.debug(f"Added '{new_category}' to categories of column '{column}'")
            df[column] = df[column].fillna(new_category)
            logger.debug(f"Filled missing values in categorical column '{column}' with '{new_category}'")
    logger.info("Completed categorical missing value imputation")
    return df


def optimize_rent_dataframe(df):
    """
    Optimize the Rent DataFrame by downcasting numerical columns and converting suitable object columns to categorical.
    """
    # Display initial memory usage
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Initial memory usage (Rent): {start_mem:.2f} MB")

    # ----------------------------
    # 1. Convert Date Columns to Datetime
    # ----------------------------
    date_columns = ['REGISTRATION_DATE', 'START_DATE', 'END_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logger.info(f"Converted column '{col}' to datetime.")

    # ----------------------------
    # 2. Convert Suitable Object Columns to Categorical
    # ----------------------------
    categorical_columns = [
        'VERSION_EN', 'AREA_EN', 'IS_FREE_HOLD_EN', 'PROP_TYPE_EN',
        'PROP_SUB_TYPE_EN', 'USAGE_EN', 'NEAREST_METRO_EN',
        'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN', 'MASTER_PROJECT_EN',
        'PROJECT_EN'
    ]

    for col in categorical_columns:
        if col in df.columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.3:
                df[col] = df[col].astype('category')
                logger.info(f"Converted '{col}' to 'category' dtype.")
            else:
                logger.info(f"Skipped converting '{col}' to 'category' dtype (cardinality ratio: {num_unique / num_total:.2f}).")

    # ----------------------------
    # 3. Downcast Integer Columns
    # ----------------------------
    int_cols = df.select_dtypes(include=['int64', 'int32', 'uint64', 'uint32']).columns.tolist()
    for col in int_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = downcast_integer(df[col])
            logger.info(f"Downcasted '{col}' from {original_dtype} to {df[col].dtype}.")

    # ----------------------------
    # 4. Downcast Float Columns
    # ----------------------------
    float_cols = df.select_dtypes(include=['float64', 'float32', 'float16']).columns.tolist()
    for col in float_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = downcast_float(df[col])
            logger.info(f"Downcasted '{col}' from {original_dtype} to {df[col].dtype}.")

    # ----------------------------
    # 5. Convert Remaining Object Columns to Categorical (Exclude 'PARKING')
    # ----------------------------
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Exclude 'PARKING' to prevent PyArrow conversion issues
    remaining_object_cols = [col for col in remaining_object_cols if col != 'PARKING']
    for col in remaining_object_cols:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.3:
            df[col] = df[col].astype('category')
            logger.info(f"Converted '{col}' to 'category' dtype.")
        else:
            logger.info(f"Skipped converting '{col}' to 'category' dtype (cardinality ratio: {num_unique / num_total:.2f}).")

    # ----------------------------
    # 6. Remove Unused Categories
    # ----------------------------
    categorical_cols = df.select_dtypes(['category']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].cat.remove_unused_categories()
        logger.info(f"Removed unused categories from '{col}'.")

    # ----------------------------
    # 7. Ensure 'PARKING' is String Type
    # ----------------------------
    if 'PARKING' in df.columns:
        df['PARKING'] = df['PARKING'].astype(str)
        logger.info("Ensured 'PARKING' column is of string type.")

    # ----------------------------
    # 8. Reset Index to RangeIndex
    # ----------------------------
    df.reset_index(drop=True, inplace=True)
    logger.info("Reset index to RangeIndex.")

    # ----------------------------
    # 9. Display Final Memory Usage
    # ----------------------------
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Final memory usage (Rent): {end_mem:.2f} MB")
    logger.info(f"Decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%")

    return df

def clean_rent_df(rent_df):
    logger.info("Starting cleaning of rent DataFrame")

    # Validate 'ACTUAL_AREA' and 'ANNUAL_AMOUNT'
    rent_df = validate_data(rent_df, ['ACTUAL_AREA', 'ANNUAL_AMOUNT'])

    # Standardize categorical columns
    categorical_columns_rent = [
        'VERSION_EN', 'AREA_EN', 'IS_FREE_HOLD_EN', 'PROP_TYPE_EN',
        'PROP_SUB_TYPE_EN', 'USAGE_EN', 'NEAREST_METRO_EN',
        'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN', 'PROJECT_EN'
    ]
    rent_df = standardize_categorical_columns(rent_df, categorical_columns_rent)

    # Add 'Unknown' category and fill missing values in categorical columns
    rent_df = fill_categorical_na(rent_df, ['PROP_SUB_TYPE_EN', 'USAGE_EN', 'PROJECT_EN'], new_category='Unknown')

    # Handle 'PARKING' if it exists
    if 'PARKING' in rent_df.columns:
        # Convert to string to ensure consistency
        rent_df['PARKING'] = rent_df['PARKING'].astype(str).str.strip().replace({'nan': 'Unknown'})
        # Optionally, convert to category if you still want categorical benefits
        # rent_df['PARKING'] = rent_df['PARKING'].astype('category')
        logger.debug("Handled 'PARKING' column by converting to string and replacing 'nan' with 'Unknown'")

    # Drop 'MASTER_PROJECT_EN' due to high missingness
    if 'MASTER_PROJECT_EN' in rent_df.columns:
        rent_df = rent_df.drop(columns=['MASTER_PROJECT_EN'])
        logger.info("Dropped column 'MASTER_PROJECT_EN'")

    # Exclude invalid 'ACTUAL_AREA' and 'ANNUAL_AMOUNT'
    initial_row_count = rent_df.shape[0]
    rent_df = rent_df[rent_df['ACTUAL_AREA'] > 0]
    rent_df = rent_df[rent_df['ANNUAL_AMOUNT'] > 0]
    logger.info(f"Excluded {initial_row_count - rent_df.shape[0]} entries with non-positive 'ACTUAL_AREA' or 'ANNUAL_AMOUNT'")

    # Count outliers
    numerical_columns = ['ANNUAL_AMOUNT', 'ACTUAL_AREA']
    rent_outliers = count_outliers(rent_df, numerical_columns)
    logger.info(f"Identified outliers in rent DataFrame: {rent_outliers}")

    # Impute missing numeric values
    rent_df = impute_missing_numeric(rent_df, strategy='median', numeric_columns=numerical_columns)

    # Optimize DataFrame
    rent_df = optimize_rent_dataframe(rent_df)

    logger.info("Completed cleaning of rent DataFrame")
    return rent_df, rent_outliers


def clean_transactions_df(transactions_df):
    logger.info("Starting cleaning of transactions DataFrame")

    # Validate 'ACTUAL_AREA' and 'TRANS_VALUE'
    transactions_df = validate_data(transactions_df, ['ACTUAL_AREA', 'TRANS_VALUE'])

    # Standardize categorical columns
    categorical_columns_transactions = [
        'GROUP_EN', 'PROCEDURE_EN', 'IS_OFFPLAN_EN', 'IS_FREE_HOLD_EN',
        'USAGE_EN', 'AREA_EN', 'PROP_TYPE_EN', 'PROP_SB_TYPE_EN',
        'PROJECT_EN'
    ]
    transactions_df = standardize_categorical_columns(transactions_df, categorical_columns_transactions)

    # Add 'Unknown' category and fill missing values in categorical columns
    transactions_df = fill_categorical_na(transactions_df, ['PROP_SB_TYPE_EN', 'PROJECT_EN'], new_category='Unknown')

    # Handle 'ROOMS_EN' if it exists
    if 'ROOMS_EN' in transactions_df.columns:
        if pd.api.types.is_categorical_dtype(transactions_df['ROOMS_EN']):
            if 'Unknown' not in transactions_df['ROOMS_EN'].cat.categories:
                transactions_df['ROOMS_EN'] = transactions_df['ROOMS_EN'].cat.add_categories(['Unknown'])
                logger.debug("Added 'Unknown' to 'ROOMS_EN' categories")
        transactions_df['ROOMS_EN'] = transactions_df['ROOMS_EN'].fillna('Unknown')
        logger.debug("Filled missing values in 'ROOMS_EN' column with 'Unknown'")

    # Handle 'PARKING' if it exists
    if 'PARKING' in transactions_df.columns:
        if pd.api.types.is_categorical_dtype(transactions_df['PARKING']):
            if 'Unknown' not in transactions_df['PARKING'].cat.categories:
                transactions_df['PARKING'] = transactions_df['PARKING'].cat.add_categories(['Unknown'])
                logger.debug("Added 'Unknown' to 'PARKING' categories")
        transactions_df['PARKING'] = transactions_df['PARKING'].fillna('Unknown')
        logger.debug("Filled missing values in 'PARKING' column with 'Unknown'")

    # Handle 'NEAREST_METRO_EN', 'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN'
    for col in ['NEAREST_METRO_EN', 'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN']:
        if col in transactions_df.columns:
            if pd.api.types.is_categorical_dtype(transactions_df[col]):
                if 'Unknown' not in transactions_df[col].cat.categories:
                    transactions_df[col] = transactions_df[col].cat.add_categories(['Unknown'])
                    logger.debug(f"Added 'Unknown' to '{col}' categories")
            transactions_df[col] = transactions_df[col].fillna('Unknown')
            logger.debug(f"Filled missing values in '{col}' column with 'Unknown'")

    # Handle 'PROCEDURE_AREA' if it exists
    if 'PROCEDURE_AREA' in transactions_df.columns:
        transactions_df['PROCEDURE_AREA'] = pd.to_numeric(transactions_df['PROCEDURE_AREA'], errors='coerce')
        transactions_df['PROCEDURE_AREA'] = transactions_df['PROCEDURE_AREA'].fillna(transactions_df['PROCEDURE_AREA'].median())
        logger.info("Handled 'PROCEDURE_AREA' missing values by filling with median")

    # Drop 'MASTER_PROJECT_EN' due to high missingness
    if 'MASTER_PROJECT_EN' in transactions_df.columns:
        transactions_df = transactions_df.drop(columns=['MASTER_PROJECT_EN'])
        logger.info("Dropped column 'MASTER_PROJECT_EN'")

    # Exclude invalid 'ACTUAL_AREA' and 'TRANS_VALUE'
    initial_row_count = transactions_df.shape[0]
    transactions_df = transactions_df[transactions_df['ACTUAL_AREA'] > 0]
    transactions_df = transactions_df[transactions_df['TRANS_VALUE'] > 0]
    logger.info(f"Excluded {initial_row_count - transactions_df.shape[0]} entries with non-positive 'ACTUAL_AREA' or 'TRANS_VALUE'")

    # Count outliers
    numerical_columns = ['TRANS_VALUE', 'ACTUAL_AREA']
    trans_outliers = count_outliers(transactions_df, numerical_columns)
    logger.info(f"Identified outliers in transactions DataFrame: {trans_outliers}")

    # Impute missing numeric values
    transactions_df = impute_missing_numeric(transactions_df, strategy='median', numeric_columns=numerical_columns)

    # Optimize DataFrame
    transactions_df = optimize_transactions_dataframe(transactions_df)

    logger.info("Completed cleaning of transactions DataFrame")
    return transactions_df, trans_outliers

@st.cache_data(show_spinner=False, ttl=3600)
def clean_and_detect_outliers(transactions, rent):
    logger.info("Starting data cleaning and outlier detection")

    # Clean DataFrames and get outlier counts
    rent_cleaned, rent_outliers = clean_rent_df(rent)
    transactions_cleaned, trans_outliers = clean_transactions_df(transactions)

    logger.info("Completed data cleaning and outlier detection")
    return transactions_cleaned, rent_cleaned, trans_outliers, rent_outliers
