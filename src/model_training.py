import pandas as pd
import logging
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_processing import preprocess_data
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess the data
try:
    logging.info("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data('data/raw/product_data.csv', 'data/processed/')
except Exception as e:
    logging.error(f"Error loading or preprocessing data: {e}")
    raise

# Create an imputer and scaler for feature scaling
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# List of models to train and evaluate
models = [
    Pipeline([('imputer', imputer), ('scaler', scaler), ('model', Ridge(alpha=0.5))]),
    Pipeline([('imputer', imputer), ('scaler', scaler), ('model', Lasso(alpha=0.1))]),
    Pipeline([('imputer', imputer), ('scaler', scaler), ('model', DecisionTreeRegressor(max_depth=5))]),
    Pipeline([('imputer', imputer), ('scaler', scaler), ('model', RandomForestRegressor(n_estimators=100, max_depth=10))]),
    Pipeline([('imputer', imputer), ('scaler', scaler), ('model', GradientBoostingRegressor(n_estimators=100, max_depth=5))])
]

# Train and evaluate models
for model in models:
    try:
        logging.info(f"Training {model['model'].__class__.__name__}...")
        model.fit(X_train, y_train)

        logging.info("Evaluating model on the testing data...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Mean Squared Error: {mse:.2f}")
        logging.info(f"Root Mean Squared Error: {rmse:.2f}")
        logging.info(f"Mean Absolute Error: {mae:.2f}")
        logging.info(f"R-squared: {r2:.2f}")

        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        logging.info(f"Cross-validation scores: {cv_scores}")
        logging.info(f"Average cross-validation score: {cv_scores.mean():.2f}")

    except Exception as e:
        logging.error(f"Error training or evaluating {model['model'].__class__.__name__}: {e}")
        continue

    # Save the trained model
    try:
        model_name = f"models/{model['model'].__class__.__name__}_price_prediction_model.pkl"
        logging.info(f"Saving trained model to {model_name}...")
        joblib.dump(model, model_name)
    except Exception as e:
        logging.error(f"Error saving trained model: {e}")