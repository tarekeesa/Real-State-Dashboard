import pandas as pd
import numpy as np

# Define downcasting functions
def downcast_integer(col):
    """
    Downcast integer columns to the smallest possible subtype without data loss.
    """
    col_min = col.min()
    col_max = col.max()
    
    if col_min >= 0:
        if col_max < 256:
            return col.astype('uint8')
        elif col_max < 65536:
            return col.astype('uint16')
        elif col_max < 4294967296:
            return col.astype('uint32')
        else:
            return col  # No downcast possible
    else:
        if col_min >= -128 and col_max <= 127:
            return col.astype('int8')
        elif col_min >= -32768 and col_max <= 32767:
            return col.astype('int16')
        elif col_min >= -2147483648 and col_max <= 2147483647:
            return col.astype('int32')
        else:
            return col  # No downcast possible

def downcast_float(col):
    """
    Downcast float columns to float32 or float16 if possible without significant loss.
    """
    # First, attempt to downcast to float32
    col_downcasted = col.astype('float32')
    
    # Check if float32 can represent the data without significant loss
    if np.allclose(col, col_downcasted, equal_nan=True):
        # Attempt to downcast to float16
        try:
            col_downcasted_16 = col_downcasted.astype('float16')
            if np.allclose(col_downcasted, col_downcasted_16, equal_nan=True):
                return col_downcasted_16
            else:
                return col_downcasted
        except:
            return col_downcasted
    else:
        return col


def further_downcast_integer(col):
    """Attempt to downcast integer columns to int8 or uint8 if possible."""
    col_min = col.min()
    col_max = col.max()
    
    if col_min >= -128 and col_max <= 127:
        return col.astype('int8')
    elif col_min >= 0 and col_max <= 255:
        return col.astype('uint8')
    else:
        return col
    

def optimize_transactions_dataframe(df):
    """
    Optimize the Transactions DataFrame by downcasting numerical columns and converting suitable object columns to categorical.
    """
    # Display initial memory usage
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Initial memory usage: {start_mem:.2f} MB")
    
    # ----------------------------
    # 1. Convert Date Columns to Datetime
    # ----------------------------
    date_columns = ['INSTANCE_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.normalize()
            print(f"Converted and normalized '{col}' to datetime.")
    
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
                print(f"Converted '{col}' to 'category' dtype.")
            else:
                print(f"Skipped converting '{col}' to 'category' dtype (cardinality ratio: {num_unique / num_total:.2f}).")
    
    # ----------------------------
    # 3. Downcast Integer Columns
    # ----------------------------
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns.tolist()
    for col in int_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = downcast_integer(df[col])
            print(f"Downcasted '{col}' from {original_dtype} to {df[col].dtype}.")
    
    # ----------------------------
    # 4. Downcast Float Columns
    # ----------------------------
    float_cols = df.select_dtypes(include=['float64']).columns.tolist()
    for col in float_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = downcast_float(df[col])
            print(f"Downcasted '{col}' from {original_dtype} to {df[col].dtype}.")
    
    # ----------------------------
    # 6. Convert Remaining Object Columns to Categorical
    # ----------------------------
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in remaining_object_cols:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.3:
            df[col] = df[col].astype('category')
            print(f"Converted '{col}' to 'category' dtype.")
        else:
            print(f"Skipped converting '{col}' to 'category' dtype (cardinality ratio: {num_unique / num_total:.2f}).")
    
    # ----------------------------
    # 7. Remove Unused Categories
    # ----------------------------
    categorical_cols = df.select_dtypes(['category']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].cat.remove_unused_categories()
        print(f"Removed unused categories from '{col}'.")
    
    # ----------------------------
    # 8. Reset Index to RangeIndex
    # ----------------------------
    df.reset_index(drop=True, inplace=True)
    print("Reset index to RangeIndex.")
    
    # ----------------------------
    # 9. Display Final Memory Usage
    # ----------------------------
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Final memory usage: {end_mem:.2f} MB")
    print(f"Decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%")
    return df
    
# Define the optimization function
def optimize_dataframe(df):
    # Display initial memory usage
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Initial memory usage: {start_mem:.2f} MB")
    
    # 1. Convert date columns to datetime
    date_columns = ['REGISTRATION_DATE', 'START_DATE', 'END_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.normalize()
            print(f"Converted and normalized '{col}' to datetime.")
    
    # 2. Convert object columns to category where appropriate
    categorical_columns = [
        'VERSION_EN', 'AREA_EN', 'IS_FREE_HOLD_EN', 'PROP_TYPE_EN',
        'PROP_SUB_TYPE_EN', 'USAGE_EN', 'NEAREST_METRO_EN',
        'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN', 'MASTER_PROJECT_EN',
        'PROJECT_EN'
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])
            # Adjusted threshold to 30% to better handle high cardinality
            if num_unique_values / num_total_values < 0.3:
                df[col] = df[col].astype('category')
                print(f"Converted '{col}' to 'category' dtype.")
            else:
                print(f"Skipped converting '{col}' to 'category' dtype (cardinality ratio: {num_unique_values / num_total_values:.2f}).")
    
    # 3. Downcast integer columns
    int_cols = df.select_dtypes(include=['int64', 'int32']).columns.tolist()
    for col in int_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = downcast_integer(df[col])
            print(f"Downcasted '{col}' from {original_dtype} to {df[col].dtype}.")
    
    # 4. Further downcast integer columns to int8 or uint8 where possible
    for col in int_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = further_downcast_integer(df[col])
            if df[col].dtype != original_dtype:
                print(f"Further downcasted '{col}' from {original_dtype} to {df[col].dtype}.")
    
    # 5. Downcast float columns
    float_cols = df.select_dtypes(include=['float64', 'float32']).columns.tolist()
    for col in float_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = downcast_float(df[col])
            print(f"Downcasted '{col}' from {original_dtype} to {df[col].dtype}.")
    
    # 6. Handle missing values in specific columns if necessary
    if 'ANNUAL_AMOUNT' in df.columns:
        original_dtype = df['ANNUAL_AMOUNT'].dtype
        df['ANNUAL_AMOUNT'] = downcast_float(df['ANNUAL_AMOUNT'])
        print(f"Downcasted 'ANNUAL_AMOUNT' from {original_dtype} to {df['ANNUAL_AMOUNT'].dtype}.")
    
    # 7. Convert remaining object columns to category if suitable
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in remaining_object_cols:
        num_unique_values = df[col].nunique()
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.3:
            df[col] = df[col].astype('category')
            print(f"Converted '{col}' to 'category' dtype.")
        else:
            print(f"Skipped converting '{col}' to 'category' dtype (cardinality ratio: {num_unique_values / num_total_values:.2f}).")
    
    # 8. Remove unused categories from categorical columns
    categorical_cols = df.select_dtypes(['category']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].cat.remove_unused_categories()
        print(f"Removed unused categories from '{col}'.")
    
    # 9. Reset index to RangeIndex
    df.reset_index(drop=True, inplace=True)
    print("Reset index to RangeIndex.")
    
    # 10. Drop unnecessary columns (if any)
    columns_to_drop = ['MASTER_PROJECT_EN']  # Add any other columns you wish to drop
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")
    
    # Display final memory usage
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Final memory usage: {end_mem:.2f} MB")
    print(f"Decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%")
    
    return df