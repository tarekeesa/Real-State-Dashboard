import pandas as pd
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def export_indexes_to_csv(*dfs, names=None):
    """
    Exports multiple DataFrames to CSV and provides download buttons.
    """
    for i, df in enumerate(dfs):
        name = names[i] if names and i < len(names) else f'DataFrame_{i}'
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f'Download {name}',
            data=csv,
            file_name=f'{name.replace(" ", "_").lower()}.csv',
            mime='text/csv'
        )
        logger.info(f"Provided download button for {name}")
