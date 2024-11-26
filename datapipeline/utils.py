# datapipeline/utils.py

import pandas as pd
import logging
import streamlit as st
import requests

from datapipeline.data_converting import downcast_float, downcast_integer

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
                # st.warning(f"Data Validation Warning: {actual_col} contains {invalid_count} zero or negative values. These rows will be excluded.")
                df = df[df[actual_col] > 0]
                logger.info(f"Excluded {invalid_count} rows with non-positive values in '{actual_col}'")
        else:
            st.warning(f"Data Validation Warning: Column '{col}' not found in DataFrame. Skipping validation for this column.")
            logger.warning(f"Column '{col}' not found in DataFrame.")
    return df


def detect_outliers_iqr(df, column):
    """
    Detects outliers in a column using the IQR method.
    Returns a boolean Series where True indicates an outlier.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    logger.debug(f"Calculated IQR for '{column}': Q1={Q1}, Q3={Q3}, IQR={IQR}, lower_bound={lower_bound}, upper_bound={upper_bound}")
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def count_outliers(df, columns):
    """
    Counts the number of outliers in specified numerical columns.
    Returns a dictionary with column names as keys and outlier counts as values.
    """
    outlier_counts = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            outliers = detect_outliers_iqr(df, col)
            count = outliers.sum()
            outlier_counts[col] = count
            logger.info(f"Detected {count} outliers in column '{col}'")
        else:
            logger.warning(f"Column '{col}' is not numeric and was skipped for outlier detection.")
    return outlier_counts


def impute_missing_values(df, strategy='median'):
    """
    Imputes missing values in the DataFrame based on the specified strategy.
    Numeric columns are imputed with median, mean, or zero.
    Categorical columns are imputed with 'Unknown'.
    """
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

def standardize_categorical_columns(df, columns, case='title'):
    """
    Standardizes specified categorical columns by stripping whitespace and adjusting case.
    """
    for col in columns:
        if col in df.columns:
            # Strip leading/trailing whitespace
            df[col] = df[col].astype(str).str.strip()
            logger.debug(f"Stripped whitespace in column '{col}'")
            # Convert case
            if case == 'upper':
                df[col] = df[col].str.upper()
            elif case == 'lower':
                df[col] = df[col].str.lower()
            elif case == 'title':
                df[col] = df[col].str.title()
            else:
                st.warning(f"Unknown case '{case}' specified for column '{col}'. No case conversion applied.")
                logger.warning(f"Unknown case '{case}' specified for column '{col}'")
    logger.info("Standardized categorical columns")
    return df



def identify_off_market(transactions_df, rent_df):
    """
    Identifies off-market properties based on the absence from both sale and rent listings.
    """
    logger.info("Identifying off-market properties")

    # Properties listed for sale and rent
    sales_properties = set(transactions_df['PROPERTY_ID'].dropna().unique())
    rent_properties = set(rent_df['PROPERTY_ID'].dropna().unique())

    # Off-market properties: present in one but not the other
    off_market_property_ids = sales_properties.symmetric_difference(rent_properties)
    logger.debug(f"Found {len(off_market_property_ids)} off-market property IDs")

    # Extract details of off-market properties from both DataFrames
    off_market_sales = transactions_df[transactions_df['PROPERTY_ID'].isin(off_market_property_ids)]
    off_market_rent = rent_df[rent_df['PROPERTY_ID'].isin(off_market_property_ids)]

    # Combine and remove duplicates
    off_market = pd.concat([off_market_sales, off_market_rent]).drop_duplicates(subset=['PROPERTY_ID'])
    logger.info(f"Total off-market properties: {off_market['PROPERTY_ID'].nunique()}")

    return off_market

def identify_distressed(transactions_df):
    """
    Identifies distressed properties based on 'PROCEDURE_EN' or 'GROUP_EN' indicators.
    """
    logger.info("Identifying distressed properties")

    # Define keywords that indicate distress
    distress_keywords = ['Mortgage', 'Delayed', 'Foreclosure', 'Repossession', 'Default']

    # Check 'PROCEDURE_EN' and 'GROUP_EN' for distress indicators
    mask_procedure = transactions_df['PROCEDURE_EN'].str.contains('|'.join(distress_keywords), case=False, na=False)
    mask_group = transactions_df['GROUP_EN'].str.contains('|'.join(distress_keywords), case=False, na=False)

    # Combine masks
    distressed_mask = mask_procedure | mask_group

    # Filter DataFrame
    distressed_properties = transactions_df[distressed_mask]
    logger.info(f"Identified {distressed_properties.shape[0]} distressed properties")

    return distressed_properties


def get_column_case_insensitive(df: pd.DataFrame, column_name: str):
    """
    Retrieves the actual column name from the DataFrame, ignoring case.

    Parameters:
    - df (pd.DataFrame): The DataFrame to search.
    - column_name (str): The column name to find, case-insensitive.

    Returns:
    - str or None: The actual column name if found, else None.
    """
    for col in df.columns:
        if col.lower() == column_name.lower():
            logger.debug(f"Found column '{col}' matching '{column_name}' (case-insensitive).")
            return col
    logger.warning(f"Column '{column_name}' not found in DataFrame columns: {df.columns.tolist()}")
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def load_data_from_api():
    logger.info("Fetching data from API")

    transactions_api_url = "https://api.yourdomain.com/transactions"
    rent_api_url = "https://api.yourdomain.com/rent"

    api_key = st.secrets["API_KEY"]  # Ensure this matches your secrets setup

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response_transactions = requests.get(transactions_api_url, headers=headers)
        response_transactions.raise_for_status()
        transactions = pd.json_normalize(response_transactions.json())

        response_rent = requests.get(rent_api_url, headers=headers)
        response_rent.raise_for_status()
        rent = pd.json_normalize(response_rent.json())

        logger.info("Data fetched successfully from API")

        transactions.columns = transactions.columns.str.strip()
        rent.columns = rent.columns.str.strip()

        return transactions, rent
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from API: {e}")
        st.error("Failed to fetch data from API. Please check the logs for more details.")
        return pd.DataFrame(), pd.DataFrame()


def standardize_column_names(df):
    """
    Standardizes column names by converting to uppercase and replacing spaces with underscores.
    """
    df.columns = df.columns.str.upper().str.replace(' ', '_')
    return df

def optimize_transactions_dataframe(df):
    """
    Optimize the Transactions DataFrame by downcasting numerical columns and converting suitable object columns to categorical.
    """
    # Display initial memory usage
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Initial memory usage: {start_mem:.2f} MB")

    # ----------------------------
    # 1. Convert Date Columns to Datetime
    # ----------------------------
    date_columns = ['INSTANCE_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.normalize()
            logger.info(f"Converted and normalized '{col}' to datetime.")

    # ----------------------------
    # 2. Convert Suitable Object Columns to Categorical
    # ----------------------------
    categorical_columns = [
        'TRANSACTION_NUMBER', 'GROUP_EN', 'PROCEDURE_EN', 'IS_OFFPLAN_EN',
        'IS_FREE_HOLD_EN', 'USAGE_EN', 'AREA_EN', 'PROP_TYPE_EN',
        'PROP_SB_TYPE_EN', 'ROOMS_EN', 'PARKING', 'NEAREST_METRO_EN',
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
    # 6. Convert Remaining Object Columns to Categorical
    # ----------------------------
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in remaining_object_cols:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.3:
            df[col] = df[col].astype('category')
            logger.info(f"Converted '{col}' to 'category' dtype.")
        else:
            logger.info(f"Skipped converting '{col}' to 'category' dtype (cardinality ratio: {num_unique / num_total:.2f}).")

    # ----------------------------
    # 7. Remove Unused Categories
    # ----------------------------
    categorical_cols = df.select_dtypes(['category']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].cat.remove_unused_categories()
        logger.info(f"Removed unused categories from '{col}'.")

    # ----------------------------
    # 8. Reset Index to RangeIndex
    # ----------------------------
    df.reset_index(drop=True, inplace=True)
    logger.info("Reset index to RangeIndex.")

    # ----------------------------
    # 9. Display Final Memory Usage
    # ----------------------------
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Final memory usage: {end_mem:.2f} MB")
    logger.info(f"Decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%")

    return df