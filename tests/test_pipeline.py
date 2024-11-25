# tests/test_pipeline.py

import unittest
import pandas as pd
import numpy as np
from datapipeline.cleaning import clean_rent_df, clean_transactions_df
from datapipeline.preprocessing import preprocess_data
from datapipeline.feature_engineering import feature_engineering
from datapipeline.index_calculation import (
    calculate_sales_supply_index,
    calculate_rental_supply_index,
    calculate_rent_vs_sales_supply_index,
    calculate_sales_asking_price_per_sqft,
    calculate_rental_asking_price_per_sqft,
    calculate_gross_yield_per_sqft
)

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        self.rent_df = pd.DataFrame({
            'REGISTRATION_DATE': pd.date_range(start='2024-01-01', periods=5, freq='M'),
            'START_DATE': pd.date_range(start='2024-01-01', periods=5, freq='M'),
            'END_DATE': pd.date_range(start='2025-01-01', periods=5, freq='M'),
            'CONTRACT_AMOUNT': [100000, 150000, 120000, 130000, 110000],
            'ANNUAL_AMOUNT': [100000, 150000, 120000, 130000, 110000],
            'ACTUAL_AREA': [1000, 1500, 1200, 1300, 1100],
            'ROOMS': [2, 3, 2, 3, 2],
            'PARKING': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'TOTAL_PROPERTIES': [50, 60, 55, 65, 70],
            'PROP_TYPE_EN': ['Unit', 'Villa', 'Unit', 'Villa', 'Unit'],
            'PROP_SUB_TYPE_EN': ['Flat', 'Villa', 'Flat', 'Villa', 'Flat'],
            'AREA_EN': ['Area1', 'Area2', 'Area1', 'Area2', 'Area1'],
            'PROJECT_EN': ['Project1', 'Project2', 'Project1', 'Project2', 'Project1'],
            'USAGE_EN': ['Residential'] * 5,
            'NEAREST_METRO_EN': ['Metro1', 'Metro2', 'Metro1', 'Metro2', 'Metro1'],
            'NEAREST_MALL_EN': ['Mall1', 'Mall2', 'Mall1', 'Mall2', 'Mall1'],
            'NEAREST_LANDMARK_EN': ['Landmark1', 'Landmark2', 'Landmark1', 'Landmark2', 'Landmark1'],
        })
        self.transactions_df = pd.DataFrame({
            'INSTANCE_DATE': pd.date_range(start='2024-01-01', periods=5, freq='M'),
            'TRANS_VALUE': [1000000, 1500000, 1200000, 1300000, 1100000],
            'ACTUAL_AREA': [1000, 1500, 1200, 1300, 1100],
            'ROOMS_EN': [2, 3, 2, 3, 2],
            'PARKING': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'TOTAL_BUYER': [1, 1, 1, 1, 1],
            'TOTAL_SELLER': [1, 1, 1, 1, 1],
            'PROP_TYPE_EN': ['Unit', 'Villa', 'Unit', 'Villa', 'Unit'],
            'PROP_SB_TYPE_EN': ['Flat', 'Villa', 'Flat', 'Villa', 'Flat'],
            'AREA_EN': ['Area1', 'Area2', 'Area1', 'Area2', 'Area1'],
            'PROJECT_EN': ['Project1', 'Project2', 'Project1', 'Project2', 'Project1'],
            'USAGE_EN': ['Residential'] * 5,
            'NEAREST_METRO_EN': ['Metro1', 'Metro2', 'Metro1', 'Metro2', 'Metro1'],
            'NEAREST_MALL_EN': ['Mall1', 'Mall2', 'Mall1', 'Mall2', 'Mall1'],
            'NEAREST_LANDMARK_EN': ['Landmark1', 'Landmark2', 'Landmark1', 'Landmark2', 'Landmark1'],
            'GROUP_EN': ['Sales'] * 5,
            'PROCEDURE_AREA': [1000, 1500, 1200, 1300, 1100],
        })

    def test_clean_rent_df(self):
        cleaned_rent_df = clean_rent_df(self.rent_df)
        self.assertFalse(cleaned_rent_df.isnull().any().any(), "Cleaned rent_df should have no missing values in critical columns.")

    def test_clean_transactions_df(self):
        cleaned_transactions_df = clean_transactions_df(self.transactions_df)
        self.assertFalse(cleaned_transactions_df.isnull().any().any(), "Cleaned transactions_df should have no missing values in critical columns.")

    def test_preprocess_data(self):
        rent_df_cleaned = clean_rent_df(self.rent_df)
        transactions_df_cleaned = clean_transactions_df(self.transactions_df)
        rent_df_processed, transactions_df_processed = preprocess_data(rent_df_cleaned, transactions_df_cleaned)
        self.assertTrue(len(rent_df_processed) > 0, "Processed rent_df should not be empty.")
        self.assertTrue(len(transactions_df_processed) > 0, "Processed transactions_df should not be empty.")

    def test_feature_engineering(self):
        rent_df_cleaned = clean_rent_df(self.rent_df)
        transactions_df_cleaned = clean_transactions_df(self.transactions_df)
        rent_df_fe, transactions_df_fe = feature_engineering(rent_df_cleaned, transactions_df_cleaned)
        self.assertIn('PROPERTY_ID', rent_df_fe.columns, "rent_df should have 'PROPERTY_ID' after feature engineering.")
        self.assertIn('PROPERTY_ID', transactions_df_fe.columns, "transactions_df should have 'PROPERTY_ID' after feature engineering.")

    def test_calculate_sales_supply_index(self):
        rent_df_cleaned = clean_rent_df(self.rent_df)
        transactions_df_cleaned = clean_transactions_df(self.transactions_df)
        rent_df_fe, transactions_df_fe = feature_engineering(rent_df_cleaned, transactions_df_cleaned)
        sales_supply_index = calculate_sales_supply_index(transactions_df_fe, rent_df_fe)
        self.assertFalse(sales_supply_index.empty, "Sales Supply Index should not be empty.")

    def test_calculate_rental_supply_index(self):
        rent_df_cleaned = clean_rent_df(self.rent_df)
        rent_df_fe, _ = feature_engineering(rent_df_cleaned, self.transactions_df)
        rental_supply_index = calculate_rental_supply_index(rent_df_fe)
        self.assertFalse(rental_supply_index.empty, "Rental Supply Index should not be empty.")

    def test_calculate_gross_yield_per_sqft(self):
        rent_df_cleaned = clean_rent_df(self.rent_df)
        transactions_df_cleaned = clean_transactions_df(self.transactions_df)
        rent_df_fe, transactions_df_fe = feature_engineering(rent_df_cleaned, transactions_df_cleaned)
        sales_price_per_sqft = calculate_sales_asking_price_per_sqft(transactions_df_fe)
        rental_price_per_sqft = calculate_rental_asking_price_per_sqft(rent_df_fe)
        gross_yield_per_sqft = calculate_gross_yield_per_sqft(sales_price_per_sqft, rental_price_per_sqft)
        self.assertFalse(gross_yield_per_sqft.empty, "Gross Yield per SqFt should not be empty.")

if __name__ == '__main__':
    unittest.main()
