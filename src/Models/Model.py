import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib  # To save the model

def convert_date_column(df):
    """
    Convert the specified date column to datetime format.
    """
    df['date_id'] = pd.to_datetime(df['date_id'], errors='coerce')
    return df

def preprocess_data(data):
    """
    Preprocess the dataset to ensure all columns are suitable for XGBoost.
    
    Parameters:
    data (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: The preprocessed dataframe.
    """
    # Convert 'date_id' to datetime format
    data = convert_date_column(data)

    # Extract time-based features from 'date_id'
    data['day'] = data['date_id'].dt.day
    data['month'] = data['date_id'].dt.month
    data['year'] = data['date_id'].dt.year
    data['day_of_week'] = data['date_id'].dt.dayofweek

    # Drop 'date_id' column as it is no longer needed
    data = data.drop(columns=['date_id'])
    
    return data

def xgboost_model(train_data):
    """
    Train the XGBoost model on the training data.

    Parameters:
    train_data (pd.DataFrame): The training data with features and target.

    Returns:
    XGBRegressor: The trained XGBoost model.
    """
    # Preprocess data to handle non-numeric columns
    train_data = preprocess_data(train_data)

    # Separate features and target variable
    X_train = train_data.drop(columns=['item_qty'])
    y_train = train_data['item_qty']
    
    # Initialize the XGBoost Regressor model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, test_data):
    """
    Evaluate the model's performance using test data.

    Parameters:
    model (XGBRegressor): The trained XGBoost model.
    test_data (pd.DataFrame): The test data with features and target.

    Returns:
    float: The Mean Absolute Error (MAE).
    float: The Root Mean Squared Error (RMSE).
    """
    # Preprocess data to handle non-numeric columns
    test_data = preprocess_data(test_data)

    # Separate features and target variable
    X_test = test_data.drop(columns=['item_qty'])
    y_test = test_data['item_qty']
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate MAE and RMSE
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    return mae, rmse

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Parameters:
    y_true (pd.Series or np.array): Actual target values.
    y_pred (pd.Series or np.array): Predicted target values.

    Returns:
    float: The MAPE value.
    """
    mape = (abs((y_true - y_pred) / y_true).mean()) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    return mape

def save_model(model, model_path):
    """
    Save the trained model to a file.

    Parameters:
    model (XGBRegressor): The trained XGBoost model.
    model_path (str): Path to save the model file.
    """
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    # Load the training and test datasets
    data_dir = 'D:/Data Science/Sem 3/Machine Learning/Course work/5/data/Processed'
    train_data_path = os.path.join(data_dir, 'train_featured_data.csv')
    test_data_path = os.path.join(data_dir, 'test_featured_data.csv')
    
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    # Train the XGBoost model
    model = xgboost_model(train_data)
    
    # Evaluate the model
    mae, rmse = evaluate_model(model, test_data)
    
    # Calculate MAPE
    X_test = preprocess_data(test_data).drop(columns=['item_qty'])
    y_test = test_data['item_qty']
    y_pred = model.predict(X_test)
    calculate_mape(y_test, y_pred)
    
    # Save the trained model
    save_model_path = os.path.join(data_dir, 'xgboost_item_qty_model.pkl')
    save_model(model, save_model_path)