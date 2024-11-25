import pandas as pd
import streamlit as st
import logging
import requests

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False, ttl=3600)
def validate_data(df, columns):
    """
    Validates that specified columns in the DataFrame have no zero or negative values.
    """
    for col in columns:
        invalid_count = (df[col] <= 0).sum()
        if invalid_count > 0:
            st.warning(f"Data Validation Warning: {col} contains zero or negative values. These rows will be excluded.")
            df = df[df[col] > 0]
            logger.info(f"Excluded {invalid_count} rows with non-positive values in '{col}'")
    return df

@st.cache_data(show_spinner=False, ttl=3600) 
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

@st.cache_data(show_spinner=False, ttl=3600)  # Cache data for 1 hour
def count_outliers(df, columns):
    """
    Counts the number of outliers in specified numerical columns.
    Returns a dictionary with column names as keys and outlier counts as values.
    """
    outlier_counts = {}
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            outliers = detect_outliers_iqr(df, col)
            count = outliers.sum()
            outlier_counts[col] = count
            logger.info(f"Detected {count} outliers in column '{col}'")
    return outlier_counts

@st.cache_data(show_spinner=False, ttl=3600)  # Cache data for 1 hour
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

@st.cache_data(show_spinner=False, ttl=3600)  # Cache data for 1 hour
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

@st.cache_data(show_spinner=False, ttl=3600)  # Cache data for 1 hour
def identify_off_market(transactions_df, rent_df):
    """
    Identifies off-market properties based on the absence from sale and rent listings.
    """
    logger.info("Identifying off-market properties")

    # Properties listed for sale and rent
    sales_properties = set(transactions_df['PROPERTY_ID'].unique())
    rent_properties = set(rent_df['PROPERTY_ID'].unique())

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

@st.cache_data(show_spinner=False, ttl=3600)  # Cache data for 1 hour
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

@st.cache_data(show_spinner=False, ttl=3600)  # Cache data for 1 hour
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

