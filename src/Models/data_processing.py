import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Function to load datasets
def load_datasets(train_path, test_path):
    """
    Load train and test data from the given file paths.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Combine train and test data
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    return combined_df

# Function to remove specific columns
def drop_unnecessary_columns(df):
    """
    Remove specified columns from the dataframe.
    """
    df_cleaned = df.drop(['item', 'invoice_num'], axis=1, errors='ignore')
    return df_cleaned

# Function to handle missing values
def drop_missing_values(df):
    """
    Identify and drop missing values from the dataframe.
    """
    missing_count = df.isnull().sum().sum()
    print(f"Total missing values: {missing_count}")
    
    # Drop rows with missing values
    df_no_missing = df.dropna()
    
    return df_no_missing

# Function to convert date columns to datetime format
def convert_date_column(df):
    """
    Convert the specified date column to datetime format.
    """
    df['date_id'] = pd.to_datetime(df['date_id'], errors='coerce')
    
    return df

# Function to perform label encoding on specified columns
def encode_categorical_columns(df):
    """
    Apply label encoding to the specified columns.
    """
    label_encoder_dept = LabelEncoder()
    df['item_dept'] = label_encoder_dept.fit_transform(df['item_dept'])
    
    label_encoder_store = LabelEncoder()
    df['store'] = label_encoder_store.fit_transform(df['store'])
    
    return df

# Function to group data by date, department, and store
def aggregate_data(df):
    """
    Group the dataframe by date, department, and store.
    """
    grouped_df = df.groupby(['date_id', 'item_dept', 'store']).sum().reset_index()
    return grouped_df

# Main processing pipeline function
def run_data_pipeline(train_path, test_path):
    """
    Full data processing pipeline.
    """
    # Step 1: Load the datasets
    combined_df = load_datasets(train_path, test_path)
    
    # Step 2: Remove unwanted columns
    df_cleaned = drop_unnecessary_columns(combined_df)
    
    # Step 3: Handle missing values
    df_no_missing = drop_missing_values(df_cleaned)
    
    # Step 4: Convert date column to datetime format
    df_converted = convert_date_column(df_no_missing)
    
    # Step 5: Label encode categorical columns
    df_encoded = encode_categorical_columns(df_converted)
    
    # Step 6: Group data by date, department, and store
    aggregated_df = aggregate_data(df_encoded)
    
    return aggregated_df

if __name__ == '__main__':
    # Define file paths for the train and test datasets
    train_path = 'D:/Data Science/Sem 3/Machine Learning/Course work/5/data/training_data.csv'
    test_path = 'D:/Data Science/Sem 3/Machine Learning/Course work/5/data/test_data.csv'
    
    # Run the data processing pipeline
    processed_data = run_data_pipeline(train_path, test_path)

    # Ensure the processed data folder exists
    output_path = 'D:/Data Science/Sem 3/Machine Learning/Course work/5/data/Processed'
    os.makedirs(output_path, exist_ok=True)

    # Save the final processed data to a CSV file
    processed_data.to_csv(os.path.join(output_path, 'final_processed_data.csv'), index=False)
    print(f"Data processing completed and saved to '{output_path}/final_processed_data.csv'.")
