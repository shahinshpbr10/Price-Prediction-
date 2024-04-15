import pandas as pd
import joblib
import sys
import logging
from flask import Flask, request, jsonify

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model
try:
    model_path = 'RandomForestRegressor_price_prediction_model.pkl'
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
    if hasattr(model, 'feature_names_in_'):
        logging.info(f"Model feature names: {model.feature_names_in_}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    sys.exit("Failed to load model, please check the model path and file.")

app = Flask(__name__)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    """
    Predict the price of a product based on the input features.
    Accepts a JSON payload with the following fields:
        category (str): The category of the product.
        demand (int): The demand for the product.
        previous_year_sales (int): The sales volume of the product in the previous year.
        previous_year_price (float): The price of the product in the previous year.
        price_change (float): Change in price from the previous year to the current year.
    Returns:
        A JSON response with the predicted price.
    """
    try:
        data = request.get_json()
        category = data['category']
        demand = data['demand']
        previous_year_sales = data['previous_year_sales']
        previous_year_price = data['previous_year_price']
        price_change = data['price_change']

        # Prepare the input features
        category_cols = ['category_clothing', 'category_electronics', 'category_furniture']
        category_values = [0, 0, 0]
        category_index = category_cols.index(f'category_{category}')
        category_values[category_index] = 1

        feature_order = ['demand', 'previous_year_sales', 'previous_year_price', 'previous_year_price_change'] + category_cols
        data_values = [demand, previous_year_sales, previous_year_price, price_change] + category_values
        input_data = pd.DataFrame([data_values], columns=feature_order)

        # Make the prediction
        predicted_price = model.predict(input_data)[0]
        logging.info(f"Predicted price: {predicted_price}")
        return jsonify({'predicted_price': predicted_price})

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({'error': 'Failed to predict price'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)