import requests

# URL of your Flask application endpoint
url = 'http://127.0.0.1:5000/predict_price'  # Replace with your actual endpoint URL

# Sample data to send in the request
data = {
    'category': 'electronics',
    'demand': 100,
    'previous_year_sales': 1000,
    'previous_year_price': 50.0,
    'price_change': 2.0
}

# Send POST request
response = requests.post(url, json=data)

# Print response
if response.status_code == 200:
    print("Prediction successful!")
    print("Predicted price:", response.json()['predicted_price'])
else:
    print("Prediction failed:", response.json()['error'])
