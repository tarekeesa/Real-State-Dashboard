from datapipeline.preprocessing import preprocess_source_data
import streamlit as st
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import gdown  
import os

from datapipeline.feature_engineering import perform_feature_engineering
from datapipeline.cleaning import clean_and_detect_outliers
from datapipeline.index_calculation import (
    calculate_gross_yield, 
    calculate_gross_yield_per_sqft, 
    calculate_rent_vs_sales_supply_index, 
    calculate_rental_asking_price, 
    calculate_rental_asking_price_per_sqft, 
    calculate_rental_supply_index, 
    calculate_sales_asking_price, 
    calculate_sales_asking_price_per_sqft, 
    calculate_sales_supply_index
)
# from datapipeline.reporting import export_indexes_to_csv
# from datapipeline.visualization import visualize_gross_yield, visualize_gross_yield_interactive, plot_gross_yield_interactive
from datapipeline.utils import load_data_from_api, identify_off_market, identify_distressed, detect_outliers_iqr, get_column_case_insensitive, standardize_column_names


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Set Streamlit page configuration
st.set_page_config(
    page_title='Dubai Real Estate Market Dashboard',
    layout='wide',
    initial_sidebar_state='expanded',
)

# -----------------------------
# Data Loading
# -----------------------------

st.sidebar.header("Data Source Selection")
data_source = st.sidebar.selectbox("Select Data Source", ["Select Data Source", "Parquet", "API"])

# Check if a valid data source is selected
if data_source == "Select Data Source":
    st.warning("Please select a data source to proceed.")
    st.stop()  # Stop further execution until a valid selection is made

# -----------------------------
# Data Loading Function
# -----------------------------
# https://drive.google.com/file/d/1d0TNyG46PwunJm1kuWhP93eQhB8eCX04/view?usp=sharing
# https://drive.google.com/file/d/1FkX2cPwvEPVZue72vEJigvFBMf-XyypG/view?usp=sharing

@st.cache_data(show_spinner=False, ttl=3600, max_entries=10)
def load_data(source='Parquet'):
    logger.info(f"Loading data from {source}")
    if source == 'Parquet':
        try:
            # Ensure the 'data' directory exists
            os.makedirs('data', exist_ok=True)
            # Define Google Drive file IDs
            transactions_file_id = '1XJjPp-lx1FxqA42s00DUGVYoKSYlScUQ'  # Replace with your actual file ID
            rent_file_id = '1SPChtZdbI9jTbavdneAcaJtVTjmoW4rm'  # Replace with your actual file ID

            # Define URLs for downloading
            transactions_url = f'https://drive.google.com/uc?id={transactions_file_id}'
            rent_url = f'https://drive.google.com/uc?id={rent_file_id}'

            #  Download the rent Parquet
            rent_path = 'data/rent.parquet'
            if not os.path.exists(rent_path):
                logger.info("Downloading rent Parquet from Google Drive...")
                gdown.download(rent_url, rent_path, quiet=False)
            else:
                logger.info("Rent Parquet already exists. Skipping download.")

            # Download the transactions Parquet
            transactions_path = 'data/transactions.parquet'
            if not os.path.exists(transactions_path):
                logger.info("Downloading transactions Parquet from Google Drive...")
                gdown.download(transactions_url, transactions_path, quiet=False)
            else:
                logger.info("Transactions Parquet already exists. Skipping download.")

            # Read the downloaded Parquet files
            rent = pd.read_parquet(rent_path)
            transactions = pd.read_parquet(transactions_path)

            # Standardize column names
            transactions = standardize_column_names(transactions)
            rent = standardize_column_names(rent)
            logger.info("Standardized column names to uppercase with underscores")

            logger.info("Data loaded successfully from Google Drive Parquet files")
        except Exception as e:
            logger.error(f"Error loading Parquet data from Google Drive: {e}")
            st.error("Failed to load data from Google Drive Parquet files. Please check the logs for more details.")
            transactions = pd.DataFrame()
            rent = pd.DataFrame()
    elif source == 'API':
        transactions, rent = load_data_from_api()
    else:
        logger.warning("Unknown data source selected")
        st.error("Unknown data source selected.")
        transactions = pd.DataFrame()
        rent = pd.DataFrame()
    
    # **Debugging Step:** Log standardized column names
    logger.info(f"Transactions DataFrame Columns (Standardized): {transactions.columns.tolist()}")
    logger.info(f"Rent DataFrame Columns (Standardized): {rent.columns.tolist()}")

    # Standardize column names: strip whitespace only
    if not transactions.empty:
        transactions.columns = transactions.columns.str.strip()
    if not rent.empty:
        rent.columns = rent.columns.str.strip()
    
    return transactions, rent


# -----------------------------
# Load Data Based on Selection
# -----------------------------

with st.spinner("Loading data, please wait..."):
    transactions_df, rent_df = load_data(source=data_source)

# -----------------------------
# Manual Data Refresh
# -----------------------------
if data_source == "API":
    if st.sidebar.button("Refresh Data"):
        # Clear the cache by calling the load_data function with a different argument
        load_data.clear()
        transactions_df, rent_df = load_data(source=data_source)
        st.success("Data refreshed successfully!")
        
# -----------------------------
# Data Cleaning and Outlier Detection
# -----------------------------

transactions_df, rent_df, trans_outliers, rent_outliers = clean_and_detect_outliers(transactions_df, rent_df)

# -----------------------------
# Preprocessing
# -----------------------------

rent_df, transactions_df = preprocess_source_data(rent_df, transactions_df)

# -----------------------------
# Feature Engineering
# -----------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def perform_feature_engineering_cached(rent, transactions):
    return perform_feature_engineering(rent, transactions)

rent_df, transactions_df = perform_feature_engineering_cached(rent_df, transactions_df)

# -----------------------------
# Index Calculation
# -----------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def calculate_all_indexes_cached(transactions, rent):
    logger.info("Starting index calculations")

    sales_supply_index = calculate_sales_supply_index(transactions, rent)
    rent_supply_index = calculate_rental_supply_index(rent)
    rent_vs_sales_supply_index = calculate_rent_vs_sales_supply_index(rent_supply_index, sales_supply_index)
    sales_asking_price_per_sqft = calculate_sales_asking_price_per_sqft(transactions)
    rental_asking_price_per_sqft = calculate_rental_asking_price_per_sqft(rent)
    sales_asking_price = calculate_sales_asking_price(transactions)
    rental_asking_price = calculate_rental_asking_price(rent)
    gross_yield_per_sqft = calculate_gross_yield_per_sqft(sales_asking_price_per_sqft, rental_asking_price_per_sqft)
    gross_yield = calculate_gross_yield(sales_asking_price, rental_asking_price)

    logger.info("Completed index calculations")
    return {
        'Sales Supply Index': sales_supply_index,
        'Rental Supply Index': rent_supply_index,
        'Rent vs Sales Supply Index': rent_vs_sales_supply_index,
        'Sales Asking Price per SqFt': sales_asking_price_per_sqft,
        'Rental Asking Price per SqFt': rental_asking_price_per_sqft,
        'Sales Asking Price': sales_asking_price,
        'Rental Asking Price': rental_asking_price,
        'Gross Yield per SqFt': gross_yield_per_sqft,
        'Gross Yield': gross_yield
    }

processed_data = calculate_all_indexes_cached(transactions_df, rent_df)

# -----------------------------
# Streamlit Dashboard Layout
# -----------------------------

# Create Tabs - 8 tabs indexed from 0 to 7
tabs = st.tabs([
    "Overview",
    "Supply Indexes",
    "Pricing Metrics",
    "Rental & Economic Indicators",  # New Tab
    "Yield Analysis",
    "Detailed Analysis",
    "Data Anomalies",
    "Data Tables",
])

# -----------------------------
# Tab 1: Overview
# -----------------------------
with tabs[0]:
    st.title('🏢 Dubai Real Estate Market Dashboard')
    st.header('Overview')
    
    # Calculate KPIs
    total_properties_for_sale = processed_data['Sales Supply Index']['Properties_Listed_For_Sale'].sum()
    total_properties_for_rent = processed_data['Rental Supply Index']['Properties_Listed_For_Rent'].sum()
    average_sales_price_per_sqft = processed_data['Sales Asking Price per SqFt']['mean_sales_price_per_sqft'].mean()
    average_rental_price_per_sqft = processed_data['Rental Asking Price per SqFt']['mean_rental_price_per_sqft'].mean()
    average_gross_yield = processed_data['Gross Yield']['Gross_Yield'].mean()
    
    # Additional KPIs
    average_transaction_value = transactions_df['TRANS_VALUE'].mean()
    total_rental_income = rent_df['CONTRACT_AMOUNT'].sum()
    
    # Handle 'ROOMS_EN' to extract numeric values for averaging
    transactions_df['Rooms_Numeric'] = transactions_df['ROOMS_EN'].str.extract('(\d+)').astype(float)
    average_rooms = transactions_df['Rooms_Numeric'].mean()
    
    # Top 10 Areas by Number of Transactions
    top_10_areas = transactions_df['AREA_EN'].value_counts().head(10).reset_index()
    top_10_areas.columns = ['Area', 'Number of Transactions']
    
    # Average Property Size
    average_property_size = transactions_df['ACTUAL_AREA'].mean()
    
    # Total Properties Listed
    total_properties_listed = total_properties_for_sale + total_properties_for_rent
    
    # Average Time on Market (For Rent)
    rent_df['START_DATE'] = pd.to_datetime(rent_df['START_DATE'])
    rent_df['END_DATE'] = pd.to_datetime(rent_df['END_DATE'])
    rent_df['Time_on_Market'] = (rent_df['END_DATE'] - rent_df['START_DATE']).dt.days
    average_time_on_market = rent_df['Time_on_Market'].mean()
    
    # Occupancy Rate
    occupancy_rate = (rent_df['TOTAL_PROPERTIES'].sum() / total_properties_listed) * 100
    
    # -----------------------------
    # Display KPIs in Multiple Rows
    # -----------------------------
    
    # Row 1: First Four KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("📈 Properties Listed for Sale", f"{total_properties_for_sale:,}")
    col2.metric("🏠 Properties Listed for Rent", f"{total_properties_for_rent:,}")
    col3.metric("💰 Avg Sales Price per SqFt", f"AED {average_sales_price_per_sqft:,.2f}")
    col4.metric("🏷️ Avg Rental Price per SqFt", f"AED {average_rental_price_per_sqft:,.2f}")
    
    # Row 2: Next Four KPIs
    col5, col6, col7, col8 = st.columns(4)
    
    col5.metric("💵 Avg Transaction Value", f"AED {average_transaction_value:,.2f}")
    col6.metric("🏢 Total Rental Income", f"AED {total_rental_income:,.2f}")
    col7.metric("🛏️ Avg Number of Rooms", f"{average_rooms:.2f}")
    col8.metric("📊 Occupancy Rate", f"{occupancy_rate:.2f}%", delta=None, delta_color="normal")
    
    st.markdown("---")
    
    # -----------------------------
    # Display Top 10 Areas by Number of Transactions
    # -----------------------------
    st.subheader("Top 10 Areas by Number of Transactions")
    
    # Rename columns for clarity
    top_10_areas.columns = ['Area', 'Number of Transactions']
    
    # Explanatory Note
    st.markdown("""
    **Note:** This chart highlights the top 10 areas with the highest number of real estate transactions. 
    Areas with more transactions indicate higher market activity and potentially better investment opportunities.
    """)
    
    # Create an interactive bar chart using Plotly Express
    fig_top_areas = px.bar(
        top_10_areas,
        x='Area',
        y='Number of Transactions',
        labels={'Area': 'Area', 'Number of Transactions': 'Number of Transactions'},
        title='Top 10 Areas with Highest Number of Transactions',
        color='Area',
        template='plotly_dark'
    )
    
    fig_top_areas.update_layout(
        xaxis_title="Area",
        yaxis_title="Number of Transactions",
        showlegend=False,
        title_x=0.5
    )
    
    st.plotly_chart(fig_top_areas, use_container_width=True)
    
    st.markdown("---")
    
    # -----------------------------
    # Additional Visualizations
    # -----------------------------
    
    # Properties Distribution by Type
    st.subheader("Properties Distribution by Type")
    
    # Explanatory Note
    st.markdown("""
    **Note:** This visualization shows the distribution of different property types in the market. 
    Understanding the proportion of each type helps in identifying market preferences and trends.
    """)
    
    prop_type_counts = transactions_df['PROP_TYPE_EN'].value_counts().reset_index()
    prop_type_counts.columns = ['Property Type', 'Count']
    
    fig = px.bar(
        prop_type_counts,
        x='Property Type',
        y='Count',
        color='Property Type',
        title='Distribution of Property Types',
        labels={'Count': 'Number of Properties', 'Property Type': 'Property Type'},
        hover_data=['Count'],
        template='plotly_dark'
    )
    fig.update_layout(
        xaxis_title="Property Type",
        yaxis_title="Number of Properties",
        showlegend=False,
        title_x=0.5
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Monthly Transactions Trend
    st.subheader("Monthly Transactions Trend")
    
    # Explanatory Note
    st.markdown("""
    **Note:** This line chart depicts the monthly trend of property transactions over the analysis period. 
    Monitoring transaction trends helps in understanding seasonal patterns and market dynamics.
    """)
    
    transactions_monthly = transactions_df.groupby('Month').size().reset_index(name='Transaction Count')
    transactions_monthly['Month'] = pd.to_datetime(transactions_monthly['Month'], format='%Y-%m')
    
    fig_transactions_trend = px.line(
        transactions_monthly,
        x='Month',
        y='Transaction Count',
        title='Monthly Property Transactions Over Time',
        labels={'Transaction Count': 'Number of Transactions', 'Month': 'Month'},
        markers=True,
        template='plotly_dark'
    )
    fig_transactions_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Transactions",
        title_x=0.5
    )
    st.plotly_chart(fig_transactions_trend, use_container_width=True)
    
    st.markdown("---")
    
    # Distribution of Property Sizes
    st.subheader("Distribution of Property Sizes (Actual Area)")
    
    # Explanatory Note
    st.markdown("""
    **Note:** This histogram illustrates the distribution of property sizes in terms of actual area (in SqFt). 
    Analyzing property sizes helps investors and buyers understand the market offerings and identify popular size ranges.
    """)
    
    fig_property_size = px.histogram(
        transactions_df,
        x='ACTUAL_AREA',
        nbins=50,
        title='Distribution of Property Sizes',
        labels={'ACTUAL_AREA': 'Actual Area (SqFt)'},
        template='plotly_dark'
    )
    fig_property_size.update_layout(
        xaxis_title="Actual Area (SqFt)",
        yaxis_title="Number of Properties",
        title_x=0.5
    )
    st.plotly_chart(fig_property_size, use_container_width=True)
    
    st.markdown("---")
    
    # Property Usage Distribution
    st.subheader("Property Usage Distribution")
    
    # Explanatory Note
    st.markdown("""
    **Note:** This pie chart represents the distribution of property usage types. 
    Understanding usage distribution aids in assessing the demand for residential, commercial, or mixed-use properties.
    """)
    
    usage_counts = transactions_df['USAGE_EN'].value_counts().reset_index()
    usage_counts.columns = ['Usage Type', 'Count']
    
    fig_usage_pie = px.pie(
        usage_counts,
        names='Usage Type',
        values='Count',
        title='Distribution of Property Usage Types',
        color='Usage Type',
        hole=0.3,
        template='plotly_dark'
    )
    fig_usage_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_usage_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Average Gross Yield by Area
    st.subheader("Average Gross Yield by Area")
    
    # Explanatory Note
    st.markdown("""
    **Note:** This bar chart showcases the top 10 and bottom 10 areas based on average gross yield. 
    Gross yield is a critical metric for investors to evaluate the return on investment in different areas.
    """)
    
    gross_yield_by_area = processed_data['Gross Yield']['Gross_Yield'].groupby(processed_data['Gross Yield']['AREA_EN']).mean().reset_index()
    top_10_yield = gross_yield_by_area.nlargest(10, 'Gross_Yield')
    bottom_10_yield = gross_yield_by_area.nsmallest(10, 'Gross_Yield')
    combined_yield = pd.concat([top_10_yield, bottom_10_yield])
    
    fig_yield_by_area = px.bar(
        combined_yield,
        x='AREA_EN',
        y='Gross_Yield',
        color='Gross_Yield',
        title='Top 10 and Bottom 10 Areas by Average Gross Yield',
        labels={'Gross_Yield': 'Average Gross Yield (%)', 'AREA_EN': 'Area'},
        template='plotly_dark',
        hover_data=['Gross_Yield']
    )
    fig_yield_by_area.update_layout(
        xaxis_title="Area",
        yaxis_title="Average Gross Yield (%)",
        showlegend=False,
        title_x=0.5
    )
    st.plotly_chart(fig_yield_by_area, use_container_width=True)
    
    st.markdown("---")

# -----------------------------
# Tab 2: Supply Indexes
# -----------------------------
with tabs[1]:
    st.header('📈 Supply Indexes')
    
    # Explanatory Note
    st.markdown("""
    **Supply Indexes** measure the availability of properties in the market. 
    Understanding supply dynamics helps in assessing market saturation and investment opportunities.
    """)
    
    # Interactive Filters
    st.sidebar.header("Supply Index Filters")
    selected_area = st.sidebar.selectbox("Select Area", options=["All"] + sorted(processed_data['Sales Supply Index']['AREA_EN'].unique()))
    selected_prop_type = st.sidebar.selectbox("Select Property Type", options=["All"] + sorted(processed_data['Sales Supply Index']['PROP_TYPE_EN'].unique()))
    
    # Filter Supply Index DataFrames based on selections
    def filter_supply_data(df, area, prop_type):
        if area != "All":
            df = df[df['AREA_EN'] == area]
        if prop_type != "All":
            df = df[df['PROP_TYPE_EN'] == prop_type]
        return df
    
    filtered_sales_supply = filter_supply_data(processed_data['Sales Supply Index'], selected_area, selected_prop_type)
    filtered_rent_supply = filter_supply_data(processed_data['Rental Supply Index'], selected_area, selected_prop_type)
    filtered_rent_vs_sales = filter_supply_data(processed_data['Rent vs Sales Supply Index'], selected_area, selected_prop_type)
    
    # Display Supply Indexes
    st.subheader("Sales Supply Index")
    st.dataframe(filtered_sales_supply[['AREA_EN', 'PROP_TYPE_EN', 'Sales_Supply_Index']])
    
    st.subheader("Rental Supply Index")
    st.dataframe(filtered_rent_supply[['AREA_EN', 'PROP_TYPE_EN', 'Rent_Supply_Index']])
    
    st.subheader("Rental vs Sales Supply Index")
    st.dataframe(filtered_rent_vs_sales[['AREA_EN', 'PROP_TYPE_EN', 'Rent_vs_Sales_Supply_Index']])
    
    # Visualization
    st.subheader("Supply Index Trends")
    
    # Explanatory Note
    st.markdown("""
    **Note:** The following line chart illustrates the trends in supply indexes across different areas and property types. 
    Monitoring these trends helps in understanding market growth and saturation levels.
    """)
    # Calculate Supply Index Trends Over Time
    # For visualization, we need to calculate the supply indexes over time (e.g., monthly)

    # First, ensure that the 'transactions_df' and 'rent_df' have 'Month' column
    # Assuming they have 'Month' column from preprocessing

    # Filtered transactions and rent data frames based on selected area and property type
    filtered_transactions = transactions_df.copy()
    filtered_rent = rent_df.copy()

    if selected_area != "All":
        filtered_transactions = filtered_transactions[filtered_transactions['AREA_EN'] == selected_area]
        filtered_rent = filtered_rent[filtered_rent['AREA_EN'] == selected_area]

    if selected_prop_type != "All":
        filtered_transactions = filtered_transactions[filtered_transactions['PROP_TYPE_EN'] == selected_prop_type]
        filtered_rent = filtered_rent[filtered_rent['PROP_TYPE_EN'] == selected_prop_type]

    # For sales supply index, we need to count the number of unique properties listed for sale per month
    sales_transactions = filtered_transactions[filtered_transactions['GROUP_EN'] == 'Sales']

    sales_counts_monthly = sales_transactions.groupby(['Month']).agg(
        Properties_Listed_For_Sale=('PROPERTY_ID', 'nunique')
    ).reset_index()

    # For rental supply index, count the number of unique properties listed for rent per month
    rental_counts_monthly = filtered_rent.groupby(['Month']).agg(
        Properties_Listed_For_Rent=('PROPERTY_ID', 'nunique')
    ).reset_index()

    # For total properties, we might need to get the total properties per month
    # Assuming 'TOTAL_PROPERTIES' is constant over time, or we can take the mean
    # Let's assume it's constant for simplicity
    if 'TOTAL_PROPERTIES' in filtered_rent.columns and not filtered_rent['TOTAL_PROPERTIES'].isnull().all():
        total_properties = filtered_rent['TOTAL_PROPERTIES'].mean()
    else:
        total_properties = 1  # To prevent division by zero if data is missing

    # Merge the counts
    supply_df = pd.merge(sales_counts_monthly, rental_counts_monthly, on='Month', how='outer').fillna(0)

    # Calculate Supply Indexes
    supply_df['Sales_Supply_Index'] = supply_df['Properties_Listed_For_Sale'] / total_properties
    supply_df['Rent_Supply_Index'] = supply_df['Properties_Listed_For_Rent'] / total_properties

    # Calculate Rent vs Sales Supply Index
    supply_df['Rent_vs_Sales_Supply_Index'] = supply_df['Properties_Listed_For_Rent'] / supply_df['Properties_Listed_For_Sale']
    supply_df['Rent_vs_Sales_Supply_Index'].replace([np.inf, -np.inf], np.nan, inplace=True)
    supply_df['Rent_vs_Sales_Supply_Index'].fillna(0, inplace=True)

    # Melt the dataframe to plot multiple lines
    supply_melted = supply_df.melt(
        id_vars='Month', 
        value_vars=['Sales_Supply_Index', 'Rent_Supply_Index', 'Rent_vs_Sales_Supply_Index'], 
        var_name='Index_Type', 
        value_name='Value'
    )

    # Convert 'Month' to datetime for proper plotting
    supply_melted['Month'] = pd.to_datetime(supply_melted['Month'])

    # Sort by 'Month' to ensure proper line plotting
    supply_melted.sort_values('Month', inplace=True)

    # Visualization using Plotly
    fig = px.line(
        supply_melted,
        x='Month',
        y='Value',
        color='Index_Type',
        title='Supply Index Trends Over Time',
        labels={'Value': 'Supply Index', 'Month': 'Month'},
        template='plotly_dark'
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Supply Index",
        title_x=0.5,
        legend_title_text='Index Type'
    )

    st.plotly_chart(fig, use_container_width=True)

    
    st.markdown("---")

# -----------------------------
# Tab 3: Pricing Metrics
# -----------------------------
with tabs[2]:
    st.header('💰 Pricing Metrics')
    
    # Explanatory Note
    st.markdown("""
    **Pricing Metrics** provide insights into the cost dynamics of properties. 
    Analyzing pricing helps buyers, sellers, and investors make informed decisions.
    """)
    
    # Interactive Filters
    st.sidebar.header("Pricing Metrics Filters")
    selected_area_p = st.sidebar.selectbox("Select Area for Pricing", options=["All"] + sorted(processed_data['Sales Asking Price per SqFt']['AREA_EN'].unique()), key='area_p')
    selected_prop_type_p = st.sidebar.selectbox("Select Property Type for Pricing", options=["All"] + sorted(processed_data['Sales Asking Price per SqFt']['PROP_TYPE_EN'].unique()), key='prop_type_p')
    
    # Filter Pricing DataFrames based on selections
    def filter_pricing_data(df, area, prop_type):
        if area != "All":
            df = df[df['AREA_EN'] == area]
        if prop_type != "All":
            df = df[df['PROP_TYPE_EN'] == prop_type]
        return df
    
    filtered_sales_price_sqft = filter_pricing_data(processed_data['Sales Asking Price per SqFt'], selected_area_p, selected_prop_type_p)
    filtered_rental_price_sqft = filter_pricing_data(processed_data['Rental Asking Price per SqFt'], selected_area_p, selected_prop_type_p)
    filtered_sales_price_aed = filter_pricing_data(processed_data['Sales Asking Price'], selected_area_p, selected_prop_type_p)
    filtered_rental_price_aed = filter_pricing_data(processed_data['Rental Asking Price'], selected_area_p, selected_prop_type_p)
    
    # Display Pricing Metrics
    st.subheader("Sales Asking Price (AED/SqFt)")
    st.dataframe(filtered_sales_price_sqft[['AREA_EN', 'PROP_TYPE_EN', 'mean_sales_price_per_sqft']])
    
    st.subheader("Rental Asking Price (AED/SqFt)")
    st.dataframe(filtered_rental_price_sqft[['AREA_EN', 'PROP_TYPE_EN', 'mean_rental_price_per_sqft']])
    
    st.subheader("Sales Asking Price (AED)")
    st.dataframe(filtered_sales_price_aed[['AREA_EN', 'PROP_TYPE_EN', 'mean_sales_price']])
    
    st.subheader("Rental Asking Price (AED)")
    st.dataframe(filtered_rental_price_aed[['AREA_EN', 'PROP_TYPE_EN', 'mean_rental_price']])
    
    # Visualization
    st.subheader("Pricing Trends")
    
    # Explanatory Note
    st.markdown("""
    **Note:** The bar chart below displays the average sales price per SqFt by area and property type. 
    This helps in comparing pricing across different segments of the market.
    """)
    
    fig4 = px.bar(
        filtered_sales_price_sqft,
        x='AREA_EN',
        y='mean_sales_price_per_sqft',
        color='PROP_TYPE_EN',
        barmode='group',
        title='Avg Sales Price per SqFt by Area and Property Type',
        template='plotly_dark'
    )
    fig4.update_layout(
        xaxis_title="Area",
        yaxis_title="Average Sales Price per SqFt (AED)",
        title_x=0.5
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Scatter Plot: Rental Price per SqFt vs Sales Price per SqFt
    st.subheader("Rental Price per SqFt vs Sales Price per SqFt")

    # Explanatory Note
    st.markdown("""
    **Note:** This scatter plot compares the average rental price per SqFt with the average sales price per SqFt for different areas. 
    This helps in identifying areas where rental yields might be higher or lower.
    """)

    # Merge the sales and rental price per SqFt data
    price_per_sqft_comparison = pd.merge(
        processed_data['Sales Asking Price per SqFt'],
        processed_data['Rental Asking Price per SqFt'],
        on=['AREA_EN', 'PROP_TYPE_EN'],
        how='inner'
    )

    # Create the scatter plot
    fig_price_comparison = px.scatter(
        price_per_sqft_comparison,
        x='mean_sales_price_per_sqft',
        y='mean_rental_price_per_sqft',
        color='PROP_TYPE_EN',
        hover_name='AREA_EN',
        labels={
            'mean_sales_price_per_sqft': 'Avg Sales Price per SqFt (AED)',
            'mean_rental_price_per_sqft': 'Avg Rental Price per SqFt (AED)'
        },
        title='Rental Price per SqFt vs Sales Price per SqFt',
        template='plotly_dark'
    )

    fig_price_comparison.update_layout(
        xaxis_title="Average Sales Price per SqFt (AED)",
        yaxis_title="Average Rental Price per SqFt (AED)",
        title_x=0.5
    )

    st.plotly_chart(fig_price_comparison, use_container_width=True)

    st.markdown("---")

    # Box Plot: Distribution of Sales Price per SqFt by Property Type
    st.subheader("Distribution of Sales Price per SqFt by Property Type")

    # Explanatory Note
    st.markdown("""
    **Note:** The box plot below shows the distribution of sales price per SqFt for different property types. 
    This visualization helps in understanding the price variation within each property type.
    """)

    fig_sales_price_box = px.box(
        processed_data['Sales Asking Price per SqFt'],
        x='PROP_TYPE_EN',
        y='mean_sales_price_per_sqft',
        color='PROP_TYPE_EN',
        title='Distribution of Sales Price per SqFt by Property Type',
        labels={
            'mean_sales_price_per_sqft': 'Avg Sales Price per SqFt (AED)',
            'PROP_TYPE_EN': 'Property Type'
        },
        template='plotly_dark'
    )

    fig_sales_price_box.update_layout(
        xaxis_title="Property Type",
        yaxis_title="Average Sales Price per SqFt (AED)",
        showlegend=False,
        title_x=0.5
    )

    st.plotly_chart(fig_sales_price_box, use_container_width=True)

    st.markdown("---")

# -----------------------------
# Tab 4: Rental & Economic Indicators (New Tab)
# -----------------------------
with tabs[3]:
    st.header('🏠 Rental & Economic Indicators')
    
    # Explanatory Note
    st.markdown("""
    **Rental & Economic Indicators** provide insights into the rental market and broader economic factors affecting real estate. 
    Understanding these indicators is crucial for assessing investment viability and market health.
    """)
    
    # Rental Asking Price Metrics
    st.subheader("Rental Asking Price Metrics")
    
    # Explanatory Note
    st.markdown("""
    **Note:** These metrics illustrate the average rental prices, both in total AED and per SqFt, across different areas and property types.
    This information helps renters and investors gauge market affordability and potential returns.
    """)
    
    # Display Rental Asking Price Metrics
    st.markdown("### Rental Asking Price (AED)")
    st.dataframe(processed_data['Rental Asking Price'][['AREA_EN', 'PROP_TYPE_EN', 'mean_rental_price']])
    
    st.markdown("### Rental Asking Price per SqFt (AED/SqFt)")
    st.dataframe(processed_data['Rental Asking Price per SqFt'][['AREA_EN', 'PROP_TYPE_EN', 'mean_rental_price_per_sqft']])
    
    # Visualization: Rental Asking Price per SqFt
    st.subheader("Rental Asking Price per SqFt by Area and Property Type")
    
    # Explanatory Note
    st.markdown("""
    **Note:** The scatter plot below visualizes the relationship between areas and their respective rental prices per SqFt across different property types. 
    This aids in identifying high-demand areas and property segments.
    """)
    
    fig_rental_price_sqft = px.scatter(
        processed_data['Rental Asking Price per SqFt'],
        x='AREA_EN',
        y='mean_rental_price_per_sqft',
        color='PROP_TYPE_EN',
        size='mean_rental_price_per_sqft',
        hover_name='PROP_TYPE_EN',
        title='Rental Asking Price per SqFt by Area and Property Type',
        template='plotly_dark'
    )
    fig_rental_price_sqft.update_layout(
        xaxis_title="Area",
        yaxis_title="Average Rental Price per SqFt (AED)",
        title_x=0.5
    )
    st.plotly_chart(fig_rental_price_sqft, use_container_width=True)
    
    st.markdown("---")
    
    # Economic Indicators (Example: Inflation Rate, Interest Rates)
    st.subheader("Economic Indicators")
    
    # Explanatory Note
    st.markdown("""
    **Note:** Economic indicators such as inflation rates and interest rates significantly impact the real estate market. 
    Tracking these indicators helps in understanding external factors influencing property prices and rental rates.
    """)
    
    # Placeholder DataFrame for Economic Indicators (Replace with actual data)
    # Assuming you have a DataFrame 'economic_df' with columns 'Indicator', 'Value', 'Date'
    # For demonstration, we'll create a sample DataFrame
    economic_df = pd.DataFrame({
        'Indicator': ['Inflation Rate', 'Interest Rate', 'GDP Growth'],
        'Value': [2.5, 3.0, 4.2],
        'Date': ['2024-09', '2024-09', '2024-09']
    })
    
    st.dataframe(economic_df)
    
    # Visualization: Economic Indicators Over Time
    st.subheader("Economic Indicators Over Time")

    # Explanatory Note
    st.markdown("""
    **Note:** The line chart below tracks economic indicators over time, providing a trend analysis that helps in forecasting real estate market movements.
    """)

    # Sample data for demonstration (Replace with actual data if available)
    economic_trends_df = pd.DataFrame({
        'Date': pd.date_range(start='2023-10-01', periods=12, freq='M'),
        'Inflation Rate': np.random.uniform(1.5, 3.5, size=12),
        'Interest Rate': np.random.uniform(2.0, 4.0, size=12),
        'GDP Growth': np.random.uniform(3.0, 5.0, size=12)
    })

    # Melt the DataFrame for plotting
    economic_trends_melted = economic_trends_df.melt(
        id_vars='Date',
        value_vars=['Inflation Rate', 'Interest Rate', 'GDP Growth'],
        var_name='Indicator',
        value_name='Value'
    )

    # Create the line chart
    fig_economic_trends = px.line(
        economic_trends_melted,
        x='Date',
        y='Value',
        color='Indicator',
        title='Economic Indicators Over Time',
        template='plotly_dark'
    )

    fig_economic_trends.update_layout(
        xaxis_title="Date",
        yaxis_title="Value (%)",
        title_x=0.5
    )

    st.plotly_chart(fig_economic_trends, use_container_width=True)

    st.markdown("---")

# -----------------------------
# Tab 5: Yield Analysis
# -----------------------------
with tabs[4]:
    st.header('📊 Yield Analysis')
    
    # Explanatory Note
    st.markdown("""
    **Yield Analysis** assesses the profitability of real estate investments by comparing rental incomes against property prices. 
    This analysis is crucial for investors to determine the return on investment and make informed purchasing decisions.
    """)
    
    # Interactive Filters
    st.sidebar.header("Yield Analysis Filters")
    selected_area_g = st.sidebar.selectbox("Select Area for Yield", options=["All"] + sorted(processed_data['Gross Yield per SqFt']['AREA_EN'].unique()), key='area_g')
    selected_prop_type_g = st.sidebar.selectbox("Select Property Type for Yield", options=["All"] + sorted(processed_data['Gross Yield per SqFt']['PROP_TYPE_EN'].unique()), key='prop_type_g')
    
    # Filter Yield DataFrames based on selections
    def filter_yield_data(df, area, prop_type):
        if area != "All":
            df = df[df['AREA_EN'] == area]
        if prop_type != "All":
            df = df[df['PROP_TYPE_EN'] == prop_type]
        return df
    
    filtered_gross_yield_sqft = filter_yield_data(processed_data['Gross Yield per SqFt'], selected_area_g, selected_prop_type_g)
    filtered_gross_yield_aed = filter_yield_data(processed_data['Gross Yield'], selected_area_g, selected_prop_type_g)
    
    # Display Gross Yield Metrics
    st.subheader("Gross Yield (%) on Asking Price (AED/SqFt)")
    st.dataframe(filtered_gross_yield_sqft[['AREA_EN', 'PROP_TYPE_EN', 'Gross_Yield_Per_SqFt']])
    
    st.subheader("Gross Yield (%) on Asking Price (AED)")
    st.dataframe(filtered_gross_yield_aed[['AREA_EN', 'PROP_TYPE_EN', 'Gross_Yield']])
    
    # Visualization
    st.subheader("Gross Yield Distribution")
    
    # Explanatory Note
    st.markdown("""
    **Note:** The scatter plot below visualizes gross yield percentages across different areas and property types. 
    Higher gross yields indicate better profitability for investments in those segments.
    """)
    
    fig5 = px.scatter(
        filtered_gross_yield_sqft,
        x='AREA_EN',
        y='Gross_Yield_Per_SqFt',
        color='PROP_TYPE_EN',
        title='Gross Yield per SqFt by Area and Property Type',
        hover_data=['Gross_Yield_Per_SqFt'],
        template='plotly_dark'
    )
    fig5.update_layout(
        xaxis_title="Area",
        yaxis_title="Gross Yield per SqFt (%)",
        title_x=0.5
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # Visualization: Rental Price Trend Over Time
    st.subheader("Rental Price Trend Over Time")

    # Explanatory Note
    st.markdown("""
    **Note:** This line chart shows the trend of average rental prices over the analysis period. 
    It helps in understanding the rental market dynamics and seasonal patterns.
    """)

    # Calculate average rental price per month
    rental_price_trend = rent_df.groupby('Month').agg(
        average_rental_price=('ANNUAL_AMOUNT', 'mean')
    ).reset_index()

    # Convert 'Month' to datetime
    rental_price_trend['Month'] = pd.to_datetime(rental_price_trend['Month'])

    # Create the line chart
    fig_rental_price_trend = px.line(
        rental_price_trend,
        x='Month',
        y='average_rental_price',
        title='Average Rental Price Over Time',
        labels={'average_rental_price': 'Average Rental Price (AED)', 'Month': 'Month'},
        markers=True,
        template='plotly_dark'
    )

    fig_rental_price_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Rental Price (AED)",
        title_x=0.5
    )

    st.plotly_chart(fig_rental_price_trend, use_container_width=True)

    st.markdown("---")

    # Visualization: Correlation between Economic Indicators and Rental Prices
    st.subheader("Correlation between Economic Indicators and Rental Prices")

    # Explanatory Note
    st.markdown("""
    **Note:** The scatter plots below illustrate the relationship between economic indicators and average rental prices over time. 
    Understanding these correlations helps in assessing how economic factors influence the rental market.
    """)

    # Merge rental price trend with economic trends
    merged_data = pd.merge(
        rental_price_trend,
        economic_trends_df,
        left_on='Month',
        right_on='Date',
        how='inner'
    )

    # Create scatter plots for each indicator
    indicators = ['Inflation Rate', 'Interest Rate', 'GDP Growth']

    for indicator in indicators:
        fig_corr = px.scatter(
            merged_data,
            x=indicator,
            y='average_rental_price',
            trendline='ols',
            title=f'Average Rental Price vs {indicator}',
            labels={'average_rental_price': 'Average Rental Price (AED)', indicator: f'{indicator} (%)'},
            template='plotly_dark'
        )
        fig_corr.update_layout(
            xaxis_title=f"{indicator} (%)",
            yaxis_title="Average Rental Price (AED)",
            title_x=0.5
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

# -----------------------------
# Tab 6: Detailed Analysis
# -----------------------------
with tabs[5]:
    st.header('🔍 Detailed Analysis')
    
    # Explanatory Note
    st.markdown("""
    **Detailed Analysis** delves deeper into specific segments of the real estate market, including off-market and distressed properties. 
    This analysis provides granular insights essential for targeted investment strategies.
    """)
    
    # Section: Off-Market Properties
    st.subheader("Off-Market Properties")
    off_market_properties = identify_off_market(transactions_df, rent_df)
    st.markdown(f"**Total Off-Market Properties:** {off_market_properties['PROPERTY_ID'].nunique()}")
    st.dataframe(off_market_properties[['PROPERTY_ID', 'AREA_EN', 'PROP_TYPE_EN', 'GROUP_EN']])
    
    # Section: Distressed Properties
    st.subheader("Distressed Properties")
    distressed_properties = identify_distressed(transactions_df)
    st.markdown(f"**Total Distressed Properties:** {distressed_properties['PROPERTY_ID'].nunique()}")
    st.dataframe(distressed_properties[['PROPERTY_ID', 'AREA_EN', 'PROP_TYPE_EN', 'PROCEDURE_EN']])
    
    # Section: Advanced Filtering and Segmentation
    st.subheader("Advanced Filtering and Segmentation")
    
    # Assign unique keys to each multiselect widget
    prop_sub_type = st.multiselect(
        "Select Property Sub-Type", 
        options=transactions_df['PROP_SB_TYPE_EN'].unique(), 
        default=transactions_df['PROP_SB_TYPE_EN'].unique(),
        key='detailed_prop_sub_type'  # Unique key
    )
    nearest_metro = st.multiselect(
        "Select Nearest Metro Station", 
        options=transactions_df['NEAREST_METRO_EN'].unique(), 
        default=transactions_df['NEAREST_METRO_EN'].unique(),
        key='detailed_nearest_metro'  # Unique key
    )
    
    # Apply Filters
    detailed_filtered = transactions_df[
        (transactions_df['PROP_SB_TYPE_EN'].isin(prop_sub_type)) &
        (transactions_df['NEAREST_METRO_EN'].isin(nearest_metro))
    ]
    
    # Debugging: Display columns and sample data
    st.write("Detailed Filtered DataFrame Columns:", detailed_filtered.columns.tolist())
    st.write("Sample Data from detailed_filtered:", detailed_filtered.head())
    st.write("Number of Rows in detailed_filtered:", detailed_filtered.shape[0])
    
    # Define columns to display
    columns_to_display = ['PROPERTY_ID', 'AREA_EN', 'PROP_TYPE_EN', 'PROP_SB_TYPE_EN', 'NEAREST_METRO_EN']
    
    # Dynamically retrieve 'Unit' column
    unit_col = get_column_case_insensitive(detailed_filtered, 'Unit')
    if unit_col:
        columns_to_display.append(unit_col)
    else:
        st.warning("'Unit' column is not available in the filtered data.")
        logger.warning("'Unit' column is missing in detailed_filtered DataFrame.")
    
    st.markdown(f"**Filtered Properties:** {detailed_filtered['PROPERTY_ID'].nunique()}")
    
    # Display the DataFrame with existing columns
    try:
        st.dataframe(detailed_filtered[columns_to_display])
    except KeyError as e:
        st.error(f"KeyError: {e}. Please check the column names.")
        logger.error(f"KeyError while displaying detailed_filtered: {e}")
    
    # Visualization of Detailed Metrics
    st.subheader("Property Distribution by Sub-Type and Type")
    
    # Explanatory Note
    st.markdown("""
    **Note:** This histogram visualizes the distribution of properties based on their sub-types and types, 
    aiding in identifying popular segments within the market.
    """)
    
    fig6 = px.histogram(
        detailed_filtered,
        x='PROP_SB_TYPE_EN',
        color='PROP_TYPE_EN',
        title='Property Distribution by Sub-Type and Type',
        labels={'PROP_SB_TYPE_EN': 'Property Sub-Type', 'count': 'Number of Properties'},
        barmode='group',
        template='plotly_dark'
    )
    st.plotly_chart(fig6, use_container_width=True)
    
    # Visualization: Procedure Types in Distressed Properties
    st.subheader("Procedure Types in Distressed Properties")
    
    # Explanatory Note
    st.markdown("""
    **Note:** This pie chart represents the distribution of different procedures involved in distressed properties. 
    Understanding these procedures is essential for investors dealing with distressed assets.
    """)
    
    if not distressed_properties.empty:
        fig7 = px.pie(
            distressed_properties,
            names='PROCEDURE_EN',
            title='Distribution of Procedures in Distressed Properties',
            hole=0.4,
            template='plotly_dark'
        )
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info('No Distressed Properties Found.')
    
    # Additional Visualization: Gross Yield Distribution by Property Type
    st.subheader("Gross Yield Distribution by Property Type")

    # Explanatory Note
    st.markdown("""
    **Note:** The box plot below shows the distribution of gross yields per SqFt across different property types. 
    This helps investors understand which property types offer higher or more consistent returns.
    """)

    fig_yield_box = px.box(
        processed_data['Gross Yield per SqFt'],
        x='PROP_TYPE_EN',
        y='Gross_Yield_Per_SqFt',
        color='PROP_TYPE_EN',
        title='Gross Yield per SqFt by Property Type',
        labels={'Gross_Yield_Per_SqFt': 'Gross Yield per SqFt (%)', 'PROP_TYPE_EN': 'Property Type'},
        template='plotly_dark'
    )

    fig_yield_box.update_layout(
        xaxis_title="Property Type",
        yaxis_title="Gross Yield per SqFt (%)",
        showlegend=False,
        title_x=0.5
    )

    st.plotly_chart(fig_yield_box, use_container_width=True)

    st.markdown("---")

    # Visualization: Heatmap of Gross Yield per SqFt by Area and Property Type
    st.subheader("Heatmap of Gross Yield per SqFt by Area and Property Type")

    # Explanatory Note
    st.markdown("""
    **Note:** The heatmap below illustrates the gross yield per SqFt across different areas and property types. 
    Darker colors indicate higher yields, highlighting potentially more profitable investment areas.
    """)

    pivot_yield = processed_data['Gross Yield per SqFt'].pivot_table(
        values='Gross_Yield_Per_SqFt',
        index='AREA_EN',
        columns='PROP_TYPE_EN',
        aggfunc='mean'
    )

    fig_yield_heatmap = px.imshow(
        pivot_yield,
        labels=dict(x="Property Type", y="Area", color="Gross Yield (%)"),
        x=pivot_yield.columns,
        y=pivot_yield.index,
        title='Heatmap of Gross Yield per SqFt by Area and Property Type',
        template='plotly_dark'
    )

    fig_yield_heatmap.update_layout(
        title_x=0.5
    )

    st.plotly_chart(fig_yield_heatmap, use_container_width=True)


    st.markdown("---")

# -----------------------------
# Tab 7: Data Anomalies
# -----------------------------
with tabs[6]:
    st.header('⚠️ Data Anomalies and Outliers')

    # Indicators for Data Anomalies
    st.subheader("Anomaly Indicators")

    # Calculate total number of data points
    total_transactions = transactions_df.shape[0]
    total_rent = rent_df.shape[0]

    # Calculate total outliers
    total_trans_outliers = sum(trans_outliers.values())
    total_rent_outliers = sum(rent_outliers.values())


    # Prevent division by zero by checking if len(trans_outliers) > 0
    if len(trans_outliers) > 0 and total_transactions > 0:
        trans_outlier_percent = (total_trans_outliers / (total_transactions * len(trans_outliers))) * 100
    else:
        trans_outlier_percent = 0
        logger.warning("Division by zero encountered while calculating trans_outlier_percent. Set to 0.")


    # Calculate percentages
    trans_outlier_percent = (total_trans_outliers / (total_transactions * len(trans_outliers))) * 100
    rent_outlier_percent = (total_rent_outliers / (total_rent * len(rent_outliers))) * 100

    # Display KPIs
    col1, col2 = st.columns(2)

    col1.metric("Transactions Data Outliers (%)", f"{trans_outlier_percent:.2f}%")
    col2.metric("Rent Data Outliers (%)", f"{rent_outlier_percent:.2f}%")

    st.markdown("---")
    # Explanatory Note
    st.markdown("""
    **Data Anomalies and Outliers** identify unusual patterns or data points that deviate significantly from the norm. 
    Detecting these anomalies is crucial for ensuring data quality and accurate analysis.
    """)
    
    # Visualization: Outlier Counts
    st.subheader("Outlier Counts by Column")
    
    # Explanatory Note
    st.markdown("""
    **Note:** The bar charts below depict the number of outliers detected in each column of the Transactions and Rent DataFrames. 
    High outlier counts may indicate data quality issues or exceptional market conditions.
    """)
    
    outlier_data = {
        "Transactions DataFrame": trans_outliers,
        "Rent DataFrame": rent_outliers
    }
    
    for df_name, counts in outlier_data.items():
        st.subheader(f"{df_name}")
        outlier_df = pd.DataFrame(list(counts.items()), columns=['Column', 'Outlier Count'])
        fig9 = px.bar(
            outlier_df,
            x='Column',
            y='Outlier Count',
            title=f'Outlier Counts in {df_name}',
            labels={'Column': 'Column', 'Outlier Count': 'Number of Outliers'},
            color='Outlier Count',
            color_continuous_scale='Reds',
            template='plotly_dark'
        )
        fig9.update_layout(
            xaxis_title="Column",
            yaxis_title="Number of Outliers",
            title_x=0.5
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    st.markdown("---")

    # Visualization: Distressed Properties Transactions Over Time
    st.subheader("Distressed Properties Transactions Over Time")

    # Explanatory Note
    st.markdown("""
    **Note:** This line chart shows the number of distressed property transactions over time. 
    Monitoring this trend can help identify periods of increased financial distress in the market.
    """)

    # Ensure 'INSTANCE_DATE' is datetime
    distressed_properties['INSTANCE_DATE'] = pd.to_datetime(distressed_properties['INSTANCE_DATE'])

    # Extract Month
    distressed_properties['Month'] = distressed_properties['INSTANCE_DATE'].dt.to_period('M').dt.to_timestamp()

    # Count distressed transactions per month
    distressed_trend = distressed_properties.groupby('Month').size().reset_index(name='Distressed Transactions')

    # Create the line chart
    fig_distressed_trend = px.line(
        distressed_trend,
        x='Month',
        y='Distressed Transactions',
        title='Distressed Property Transactions Over Time',
        labels={'Distressed Transactions': 'Number of Transactions', 'Month': 'Month'},
        markers=True,
        template='plotly_dark'
    )

    fig_distressed_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Distressed Transactions",
        title_x=0.5
    )

    st.plotly_chart(fig_distressed_trend, use_container_width=True)

    st.markdown("---")
    # Visualization: Top Areas with Distressed Properties
    st.subheader("Top Areas with Distressed Properties")

    # Explanatory Note
    st.markdown("""
    **Note:** This bar chart displays the areas with the highest number of distressed properties. 
    Identifying these areas helps investors focus on locations with potential discounted opportunities.
    """)

    top_distressed_areas = distressed_properties['AREA_EN'].value_counts().head(10).reset_index()
    top_distressed_areas.columns = ['Area', 'Number of Distressed Properties']

    fig_top_distressed_areas = px.bar(
        top_distressed_areas,
        x='Area',
        y='Number of Distressed Properties',
        color='Area',
        title='Top 10 Areas with Distressed Properties',
        labels={'Number of Distressed Properties': 'Number of Distressed Properties', 'Area': 'Area'},
        template='plotly_dark'
    )

    fig_top_distressed_areas.update_layout(
        xaxis_title="Area",
        yaxis_title="Number of Distressed Properties",
        showlegend=False,
        title_x=0.5
    )

    st.plotly_chart(fig_top_distressed_areas, use_container_width=True)

    st.markdown("---")
    # Visualization: Distribution of Transaction Values
    st.subheader("Distribution of Transaction Values")

    # Explanatory Note
    st.markdown("""
    **Note:** This histogram shows the distribution of transaction values, highlighting any unusually high or low values that may be outliers.
    """)

    fig_trans_value_hist = px.histogram(
        transactions_df,
        x='TRANS_VALUE',
        nbins=100,
        title='Distribution of Transaction Values',
        labels={'TRANS_VALUE': 'Transaction Value (AED)'},
        template='plotly_dark'
    )

    fig_trans_value_hist.update_layout(
        xaxis_title="Transaction Value (AED)",
        yaxis_title="Count",
        title_x=0.5
    )

    st.plotly_chart(fig_trans_value_hist, use_container_width=True)

    st.markdown("---")
    # Visualization: Box Plot of Actual Area
    st.subheader("Box Plot of Actual Area")

    # Explanatory Note
    st.markdown("""
    **Note:** The box plot below displays the distribution of 'Actual Area' in both Transactions and Rent DataFrames, helping to visualize outliers and data spread.
    """)

    # Combine data for plotting
    area_data = pd.DataFrame({
        'Actual Area': pd.concat([transactions_df['ACTUAL_AREA'], rent_df['ACTUAL_AREA']], ignore_index=True),
        'Dataset': ['Transactions'] * len(transactions_df) + ['Rent'] * len(rent_df)
    })

    fig_area_box = px.box(
        area_data,
        x='Dataset',
        y='Actual Area',
        color='Dataset',
        title='Box Plot of Actual Area in Transactions and Rent Data',
        template='plotly_dark'
    )

    fig_area_box.update_layout(
        xaxis_title="Dataset",
        yaxis_title="Actual Area (SqFt)",
        showlegend=False,
        title_x=0.5
    )

    st.plotly_chart(fig_area_box, use_container_width=True)

    st.markdown("---")
    # Visualization: Anomalies Over Time
    st.subheader("Anomalies Over Time")

    # Explanatory Note
    st.markdown("""
    **Note:** This line chart shows the number of detected outliers over time, helping to identify periods with increased data anomalies.
    """)

    # Detect outliers over time for Transactions
    transactions_df['Outlier_TRANS_VALUE'] = detect_outliers_iqr(transactions_df, 'TRANS_VALUE')
    transactions_df['Month'] = pd.to_datetime(transactions_df['INSTANCE_DATE']).dt.to_period('M').dt.to_timestamp()
    outliers_over_time = transactions_df.groupby('Month')['Outlier_TRANS_VALUE'].sum().reset_index()

    # Create the line chart
    fig_outliers_time = px.line(
        outliers_over_time,
        x='Month',
        y='Outlier_TRANS_VALUE',
        title='Number of Transaction Value Outliers Over Time',
        labels={'Outlier_TRANS_VALUE': 'Number of Outliers', 'Month': 'Month'},
        markers=True,
        template='plotly_dark'
    )

    fig_outliers_time.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Outliers",
        title_x=0.5
    )

    st.plotly_chart(fig_outliers_time, use_container_width=True)

    st.markdown("---")

# -----------------------------
# Tab 8: Data Tables
# -----------------------------
with tabs[7]:
    st.header('📊 Detailed Data Tables')
    
    # Explanatory Note
    st.markdown("""
    **Detailed Data Tables** allow users to explore the underlying data used for analysis. 
    Reviewing raw data is essential for transparency and validating the dashboard's insights.
    """)
    
    # Select Data to View
    st.subheader("Processed Data")
    data_option = st.selectbox('Select Processed Data to View', list(processed_data.keys()))
    st.dataframe(processed_data[data_option])
    
    # Source Data
    st.subheader('Source Data')
    source_option = st.selectbox('Select Source Data to View', ['Rent Data', 'Transactions Data'])
    if source_option == 'Rent Data':
        st.dataframe(rent_df)
    else:
        st.dataframe(transactions_df)
    
    st.markdown("---")
    
    # Download Processed Data
st.subheader('💾 Download Processed Data')
for name, df in processed_data.items():
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f'Download {name}',
        data=csv,
        file_name=f'{name.replace(" ", "_").lower()}.csv',
        mime='text/csv',
        key=f'download_processed_{name}'  # Unique key
    )
    
    st.markdown("---")

# -----------------------------
# Footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.write("Developed by Tarek Eissa")

st.write("""
*This dashboard provides an interactive overview of the Dubai real estate market, allowing investors and stakeholders to explore key metrics, compare different areas and property types, and gain insights into market trends and opportunities.*
""")
