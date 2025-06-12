import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_and_save_model():
    # Step 1: Load data
    data_path = os.path.join('data', 'house_data.csv')
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: '{data_path}' not found. Please ensure the data file is in the data/ directory.")
        return

    # Step 2: Select features and target variable
    # Adjust these columns depending on your dataset
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'LotArea', 'Neighborhood']
    target = 'SalePrice']

    # Remove rows with NaN values in important columns (customize)
    df.dropna(subset=features + [target], inplace=True)

    X = df[features]
    y = df[target]

    # Step 3: Identify numerical and categorical columns
    numeric_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'LotArea']
    categorical_features = ['Neighborhood']

    # Step 4: Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Step 5: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Create a pipeline combining preprocessing and the XGBoost model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', XGBRegressor(random_state=42))])

    # Step 7: Train the model
    print("Training model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Step 8: Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel evaluation on the test set:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # Step 9: Save the trained model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True) # Create models directory if it doesn't exist
    model_path = os.path.join(model_dir, 'house_price_predictor.pkl')
    joblib.dump(model_pipeline, model_path)
    print(f"\nTrained and saved the model at '{model_path}'")

if __name__ == "__main__":
    train_and_save_model()