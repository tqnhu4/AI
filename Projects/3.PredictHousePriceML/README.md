# House Price Prediction Project

This interactive web application uses Machine Learning to predict house prices based on input features.

## Technologies Used

* **Python**
* **Pandas**: Data manipulation and analysis.
* **Scikit-learn**: Data preprocessing and ML Pipeline building.
* **XGBoost**: Machine Learning algorithm for regression.
* **Streamlit**: Building the web user interface.
* **joblib**: Saving and loading models.

## Project Structure

```text
house_price_predictor/
├── data/
│   └── house_data.csv
├── models/
│   └── house_price_predictor.pkl
├── scripts/
│   ├── train_model.py
│   └── preprocess_data.py  (Optional)
├── app.py
├── requirements.txt
└── README.md
```

## Environment Setup

1.  **Clone this repository (if applicable)** or create the folder structure as above.
2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```
3.  **Install required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

1.  Download a house price dataset (e.g., "House Prices - Advanced Regression Techniques" from Kaggle).
2.  Place the downloaded CSV file (rename it to `house_data.csv` if necessary) into the `data/` directory.

## Model Training

Run the model training script to preprocess the data and train the Machine Learning model. The trained model will be saved to `models/house_price_predictor.pkl`.

```bash
python scripts/train_model.py
```

Note: If you have a separate preprocess_data.py file, run it before train_model.py.

## Running the Web Application
Once the model is trained and saved, you can launch the Streamlit application:

```text
streamlit run app.py
```

The application will open in your web browser (usually at http://localhost:8501).

