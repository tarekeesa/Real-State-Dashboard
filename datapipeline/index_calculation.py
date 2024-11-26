# datapipeline/index_calculation.py

import pandas as pd
import numpy as np
import logging
import streamlit as st

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=False)
def calculate_sales_supply_index(transactions_df, rent_df):
    logger.info("Calculating Sales Supply Index")

    # Filter sales transactions
    sales_transactions = transactions_df[transactions_df['GROUP_EN'] == 'Sales']
    logger.debug(f"Filtered sales transactions: {sales_transactions.shape[0]} records")

    # Count properties listed for sale
    sales_counts = sales_transactions.groupby(['AREA_EN', 'PROP_TYPE_EN']).agg(
        Properties_Listed_For_Sale=('PROPERTY_ID', 'nunique')
    ).reset_index()
    logger.debug("Calculated properties listed for sale")

    # Calculate total properties from rent_df
    total_properties = rent_df.groupby(['AREA_EN', 'PROP_TYPE_EN']).agg(
        TOTAL_PROPERTIES=('TOTAL_PROPERTIES', 'mean')
    ).reset_index()
    logger.debug("Calculated total properties from rent DataFrame")

    # Merge and calculate Sales Supply Index
    sales_supply_index = pd.merge(sales_counts, total_properties, on=['AREA_EN', 'PROP_TYPE_EN'], how='left')
    logger.debug("Merged sales counts with total properties")

    # Exclude entries where 'TOTAL_PROPERTIES' <= 0 to prevent division by zero
    initial_row_count = sales_supply_index.shape[0]
    sales_supply_index = sales_supply_index[sales_supply_index['TOTAL_PROPERTIES'] > 0]
    logger.info(f"Excluded {initial_row_count - sales_supply_index.shape[0]} entries with non-positive 'TOTAL_PROPERTIES'")

    sales_supply_index['Sales_Supply_Index'] = sales_supply_index['Properties_Listed_For_Sale'] / sales_supply_index['TOTAL_PROPERTIES']
    logger.info("Calculated Sales Supply Index")

    return sales_supply_index

@st.cache_data(show_spinner=False)
def calculate_rental_supply_index(rent_df):
    logger.info("Calculating Rental Supply Index")

    # Count properties listed for rent
    rental_counts = rent_df.groupby(['AREA_EN', 'PROP_TYPE_EN']).agg(
        Properties_Listed_For_Rent=('PROPERTY_ID', 'nunique')
    ).reset_index()
    logger.debug("Calculated properties listed for rent")

    # Calculate total properties
    total_properties = rent_df.groupby(['AREA_EN', 'PROP_TYPE_EN']).agg(
        TOTAL_PROPERTIES=('TOTAL_PROPERTIES', 'mean')
    ).reset_index()
    logger.debug("Calculated total properties from rent DataFrame")

    # Merge and calculate Rent Supply Index
    rent_supply_index = pd.merge(rental_counts, total_properties, on=['AREA_EN', 'PROP_TYPE_EN'], how='left')
    logger.debug("Merged rental counts with total properties")

    # Exclude entries where 'TOTAL_PROPERTIES' <= 0 to prevent division by zero
    initial_row_count = rent_supply_index.shape[0]
    rent_supply_index = rent_supply_index[rent_supply_index['TOTAL_PROPERTIES'] > 0]
    logger.info(f"Excluded {initial_row_count - rent_supply_index.shape[0]} entries with non-positive 'TOTAL_PROPERTIES'")

    rent_supply_index['Rent_Supply_Index'] = rent_supply_index['Properties_Listed_For_Rent'] / rent_supply_index['TOTAL_PROPERTIES']
    logger.info("Calculated Rent Supply Index")

    return rent_supply_index

@st.cache_data(show_spinner=False)
def calculate_rent_vs_sales_supply_index(rent_supply_index, sales_supply_index):
    logger.info("Calculating Rent vs Sales Supply Index")

    # Merge rental and sales counts
    rent_vs_sales = pd.merge(rent_supply_index, sales_supply_index, on=['AREA_EN', 'PROP_TYPE_EN'], how='inner')
    logger.debug("Merged rent supply index with sales supply index")

    # Exclude entries where 'Properties_Listed_For_Sale' <= 0 to prevent division by zero
    initial_row_count = rent_vs_sales.shape[0]
    rent_vs_sales = rent_vs_sales[rent_vs_sales['Properties_Listed_For_Sale'] > 0]
    logger.info(f"Excluded {initial_row_count - rent_vs_sales.shape[0]} entries with non-positive 'Properties_Listed_For_Sale'")

    # Calculate Rent vs Sales Supply Index
    rent_vs_sales['Rent_vs_Sales_Supply_Index'] = rent_vs_sales['Properties_Listed_For_Rent'] / rent_vs_sales['Properties_Listed_For_Sale']
    rent_vs_sales['Rent_vs_Sales_Supply_Index'].replace([np.inf, -np.inf], np.nan, inplace=True)
    rent_vs_sales['Rent_vs_Sales_Supply_Index'].fillna(0, inplace=True)
    logger.info("Calculated Rent vs Sales Supply Index")

    return rent_vs_sales

@st.cache_data(show_spinner=False)
def calculate_sales_asking_price_per_sqft(transactions_df):
    logger.info("Calculating Sales Asking Price per SqFt")

    # Calculate Sales Asking Price per SqFt
    transactions_df['Sales_Price_per_SqFt'] = transactions_df['TRANS_VALUE'] / transactions_df['ACTUAL_AREA']
    logger.debug("Calculated 'Sales_Price_per_SqFt'")

    # Aggregate by desired levels
    sales_price_stats = transactions_df.groupby(['AREA_EN', 'PROP_TYPE_EN']).agg(
        mean_sales_price_per_sqft=('Sales_Price_per_SqFt', 'mean'),
        median_sales_price_per_sqft=('Sales_Price_per_SqFt', 'median'),
        std_sales_price_per_sqft=('Sales_Price_per_SqFt', 'std')
    ).reset_index()
    logger.debug("Aggregated sales price statistics")

    # Exclude entries where 'mean_sales_price_per_sqft' <= 0
    initial_row_count = sales_price_stats.shape[0]
    sales_price_stats = sales_price_stats[sales_price_stats['mean_sales_price_per_sqft'] > 0]
    logger.info(f"Excluded {initial_row_count - sales_price_stats.shape[0]} entries with non-positive 'mean_sales_price_per_sqft'")

    return sales_price_stats

@st.cache_data(show_spinner=False)
def calculate_rental_asking_price_per_sqft(rent_df):
    logger.info("Calculating Rental Asking Price per SqFt")

    # Calculate Rental Asking Price per SqFt
    rent_df['Rental_Price_per_SqFt'] = rent_df['ANNUAL_AMOUNT'] / rent_df['ACTUAL_AREA']
    logger.debug("Calculated 'Rental_Price_per_SqFt'")

    # Aggregate by desired levels
    rental_price_stats = rent_df.groupby(['AREA_EN', 'PROP_TYPE_EN']).agg(
        mean_rental_price_per_sqft=('Rental_Price_per_SqFt', 'mean'),
        median_rental_price_per_sqft=('Rental_Price_per_SqFt', 'median'),
        std_rental_price_per_sqft=('Rental_Price_per_SqFt', 'std')
    ).reset_index()
    logger.debug("Aggregated rental price statistics")

    return rental_price_stats

@st.cache_data(show_spinner=False)
def calculate_sales_asking_price(transactions_df):
    logger.info("Calculating Sales Asking Price")

    # Aggregate Sales Asking Price
    sales_price = transactions_df.groupby(['AREA_EN', 'PROP_TYPE_EN']).agg(
        mean_sales_price=('TRANS_VALUE', 'mean'),
        median_sales_price=('TRANS_VALUE', 'median'),
        std_sales_price=('TRANS_VALUE', 'std')
    ).reset_index()
    logger.debug("Aggregated sales price statistics")

    # Exclude entries where 'mean_sales_price' <= 0
    initial_row_count = sales_price.shape[0]
    sales_price = sales_price[sales_price['mean_sales_price'] > 0]
    logger.info(f"Excluded {initial_row_count - sales_price.shape[0]} entries with non-positive 'mean_sales_price'")

    return sales_price

@st.cache_data(show_spinner=False)
def calculate_rental_asking_price(rent_df):
    logger.info("Calculating Rental Asking Price")

    # Aggregate Rental Asking Price
    rental_price = rent_df.groupby(['AREA_EN', 'PROP_TYPE_EN']).agg(
        mean_rental_price=('ANNUAL_AMOUNT', 'mean'),
        median_rental_price=('ANNUAL_AMOUNT', 'median'),
        std_rental_price=('ANNUAL_AMOUNT', 'std')
    ).reset_index()
    logger.debug("Aggregated rental price statistics")

    return rental_price

@st.cache_data(show_spinner=False)
def calculate_gross_yield_per_sqft(sales_asking_price_per_sqft, rental_asking_price_per_sqft):
    logger.info("Calculating Gross Yield per SqFt")

    # Merge sales and rental price per SqFt
    price_per_sqft = pd.merge(
        sales_asking_price_per_sqft,
        rental_asking_price_per_sqft,
        on=['AREA_EN', 'PROP_TYPE_EN'],
        how='inner'
    )
    logger.debug("Merged sales and rental price per SqFt")

    # Exclude or handle zero 'mean_sales_price_per_sqft' to prevent division by zero
    initial_row_count = price_per_sqft.shape[0]
    price_per_sqft = price_per_sqft[price_per_sqft['mean_sales_price_per_sqft'] > 0]
    logger.info(f"Excluded {initial_row_count - price_per_sqft.shape[0]} entries with non-positive 'mean_sales_price_per_sqft'")

    # Calculate Gross Yield (%)
    price_per_sqft['Gross_Yield_Per_SqFt'] = (
        price_per_sqft['mean_rental_price_per_sqft'] /
        price_per_sqft['mean_sales_price_per_sqft']
    ) * 100
    logger.debug("Calculated 'Gross_Yield_Per_SqFt'")

    # Replace infinite or undefined values with NaN
    price_per_sqft['Gross_Yield_Per_SqFt'] = price_per_sqft['Gross_Yield_Per_SqFt'].replace([np.inf, -np.inf], np.nan)

    # Impute NaN values with median Gross Yield
    median_gross_yield = price_per_sqft['Gross_Yield_Per_SqFt'].median()
    price_per_sqft['Gross_Yield_Per_SqFt'] = price_per_sqft['Gross_Yield_Per_SqFt'].fillna(median_gross_yield)
    logger.info("Handled NaN values in 'Gross_Yield_Per_SqFt' by imputing with median")

    return price_per_sqft[['AREA_EN', 'PROP_TYPE_EN', 'Gross_Yield_Per_SqFt']]

@st.cache_data(show_spinner=False)
def calculate_gross_yield(sales_asking_price, rental_asking_price):
    logger.info("Calculating Gross Yield")

    # Merge sales and rental prices
    price = pd.merge(
        sales_asking_price,
        rental_asking_price,
        on=['AREA_EN', 'PROP_TYPE_EN'],
        how='inner'
    )
    logger.debug("Merged sales and rental prices")

    # Exclude or handle zero 'mean_sales_price' to prevent division by zero
    initial_row_count = price.shape[0]
    price = price[price['mean_sales_price'] > 0]
    logger.info(f"Excluded {initial_row_count - price.shape[0]} entries with non-positive 'mean_sales_price'")

    # Calculate Gross Yield (%)
    price['Gross_Yield'] = (
        price['mean_rental_price'] /
        price['mean_sales_price']
    ) * 100
    logger.debug("Calculated 'Gross_Yield'")

    # Replace infinite or undefined values with NaN
    price['Gross_Yield'] = price['Gross_Yield'].replace([np.inf, -np.inf], np.nan)

    # Impute NaN values with median Gross Yield
    median_gross_yield = price['Gross_Yield'].median()
    price['Gross_Yield'] = price['Gross_Yield'].fillna(median_gross_yield)
    logger.info("Handled NaN values in 'Gross_Yield' by imputing with median")

    return price[['AREA_EN', 'PROP_TYPE_EN', 'Gross_Yield']]
