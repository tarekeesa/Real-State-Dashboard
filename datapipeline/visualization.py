import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
import logging

logger = logging.getLogger(__name__)

def visualize_gross_yield(gross_yield_per_sqft_df):
    logger.info("Visualizing Gross Yield per SqFt using boxplot")

    # Boxplot of Gross Yield per SqFt by Area
    fig, ax = plt.subplots(figsize=(10, 6))
    gross_yield_per_sqft_df.boxplot(
        column='Gross_Yield_Per_SqFt',
        by='AREA_EN',
        ax=ax,
        grid=False
    )
    ax.set_title('Gross Yield per SqFt by Area')
    ax.set_xlabel('Area')
    ax.set_ylabel('Gross Yield (%)')
    plt.suptitle('')  # Suppress the automatic 'Boxplot grouped by AREA_EN' title

    # Use Streamlit to display the plot
    st.pyplot(fig)
    logger.info("Displayed boxplot of Gross Yield per SqFt")

def visualize_gross_yield_interactive(gross_yield_df):
    logger.info("Visualizing Gross Yield per SqFt using interactive scatter plot")

    fig = px.scatter(
        gross_yield_df,
        x='AREA_EN',
        y='Gross_Yield_Per_SqFt',
        color='PROP_TYPE_EN',
        title='Gross Yield per SqFt by Area and Property Type',
        hover_data=['Gross_Yield_Per_SqFt']
    )
    fig.update_layout(xaxis_title='Area', yaxis_title='Gross Yield (%)')
    st.plotly_chart(fig, use_container_width=True)
    logger.info("Displayed interactive scatter plot of Gross Yield per SqFt")

def plot_gross_yield_interactive(gross_yield_df, level="Area"):
    logger.info(f"Visualizing Gross Yield per SqFt by {level}")

    fig = px.scatter(
        gross_yield_df,
        x=level,
        y='Gross_Yield_Per_SqFt',
        color='PROP_TYPE_EN',
        title=f'Gross Yield per SqFt by {level}',
        hover_data=['Gross_Yield_Per_SqFt']
    )
    st.plotly_chart(fig, use_container_width=True)
    logger.info(f"Displayed interactive scatter plot of Gross Yield per SqFt by {level}")
