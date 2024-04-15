import pandas as pd
import numpy as np

# Define the original dataset
data = {
    "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "category": ["electronics", "clothing", "furniture", "electronics", "clothing", "furniture",
                 "electronics", "clothing", "furniture", "electronics"],
    "demand": [150, 80, 30, 120, 90, 45, 180, 75, 35, 100],
    "previous_year_sales": [1000, 600, 200, 800, 700, 300, 1200, 550, 250, 900],
    "previous_year_price": [99.99, 29.99, 149.99, 89.99, 24.99, 199.99, 119.99, 34.99, 179.99, 79.99],
    "price": [109.99, 34.99, 179.99, 99.99, 29.99, 239.99, 129.99, 39.99, 209.99, 89.99]
}
df = pd.DataFrame(data)

# Define realistic ranges for adjustments
demand_variation = 20  # Demand fluctuation up to +/- 20 units
sales_variation = 100  # Sales fluctuation up to +/- 100 units
price_fluctuation_percentage = 0.05  # Prices fluctuate up to +/- 5%

# Number of additional entries needed
num_additional_entries = 1490

# Sample from the existing data and modify to create new entries
np.random.seed(42)  # For reproducibility
new_entries = df.sample(n=num_additional_entries, replace=True).reset_index(drop=True)
new_entries['product_id'] = range(11, 11 + num_additional_entries)
new_entries['demand'] += np.random.randint(-demand_variation, demand_variation + 1, size=num_additional_entries)
new_entries['previous_year_sales'] += np.random.randint(-sales_variation, sales_variation + 1, size=num_additional_entries)
new_entries['previous_year_price'] *= (1 + np.random.uniform(-price_fluctuation_percentage, price_fluctuation_percentage, size=num_additional_entries))
new_entries['price'] *= (1 + np.random.uniform(-price_fluctuation_percentage, price_fluctuation_percentage, size=num_additional_entries))

# Correcting the format to ensure no extreme values and maintaining format
new_entries['previous_year_price'] = new_entries['previous_year_price'].round(2)
new_entries['price'] = new_entries['price'].round(2)

# Combine the original and new data into a single DataFrame
expanded_df = pd.concat([df, new_entries]).reset_index(drop=True)

# Save the expanded dataset to a CSV file
expanded_df.to_csv('expanded_data.csv', index=False)

# Print sample to verify the data
print(expanded_df.head(15))  # Display the first 15 rows to check the variety and consistency
