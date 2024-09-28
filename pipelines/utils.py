import logging
import pandas as pd

from src.data_cleaning import DataCleaning, DataPreProcessStratrgy


def get_data_for_test():
    try:
        # Corrected the method name to read_csv (lowercase 'csv')
        df = pd.read_csv("../data/olist_customers_dataset.csv")
        
        # Sample 100 rows
        df = df.sample(n=100)
        
        # Apply the data preprocessing strategy
        preprocess_strategy = DataPreProcessStratrgy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        
        # Drop the 'review_score' column
        df.drop(["review_score"], axis=1, inplace=True)
        
        # Convert DataFrame to JSON format
        result = df.to_json(orient="split")
        
        return result
    except Exception as e:
        logging.error(f"Error in get_data_for_test: {e}")
        raise