import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(data_file, output_dir):
    """
    Preprocess the product data for price prediction.

    Args:
        data_file (str): Path to the raw data file.
        output_dir (str): Path to the directory where preprocessed data should be saved.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load the data
    data = pd.read_csv(data_file)

    # Handle missing values
    data['demand'] = data['demand'].fillna(data['demand'].mean())
    data['previous_year_sales'] = data['previous_year_sales'].fillna(data['previous_year_sales'].mean())
    data['previous_year_price'] = data['previous_year_price'].fillna(data['previous_year_price'].mean())

    # Create additional features
    data['previous_year_price_change'] = (data['previous_year_price'] - data['previous_year_price'].shift(1)) / data['previous_year_price'].shift(1)

    # Encode categorical features
    data = pd.get_dummies(data, columns=['category'])

    # Split the data into training and testing sets
    X = data[['demand', 'previous_year_sales', 'previous_year_price', 'previous_year_price_change'] + [col for col in data.columns if col.startswith('category_')]]
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessed data
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False, header=True)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False, header=True)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test = preprocess_data('data/raw/product_data.csv', 'data/processed/')