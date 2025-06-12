import streamlit as st
import pandas as pd
import joblib
import os

# Path to the trained model
model_path = os.path.join('models', 'house_price_predictor.pkl')

# Load the trained model
@st.cache_resource # Cache the model so it doesn't reload with every interaction
def load_model():
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Please ensure you have run 'python scripts/train_model.py' to train and save the model.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

st.title("🏡 House Price Predictor")

st.write("""
This application uses a Machine Learning model to predict house prices based on the features you provide.
""")

st.sidebar.header("Enter House Features")

# Collect user input features
def user_input_features():
    gr_liv_area = st.sidebar.slider("Living Area (sq ft)", 500, 5000, 1500)
    bedroom_abv_gr = st.sidebar.slider("Bedrooms Above Ground", 0, 8, 3)
    full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
    year_built = st.sidebar.slider("Year Built", 1900, 2025, 2000)
    lot_area = st.sidebar.slider("Lot Area (sq ft)", 1000, 100000, 10000)

    # Get list of neighborhoods from the trained model (if possible) or hardcode
    # Ideally, this should come from the encoded columns in the model_pipeline's preprocessor
    # For simplicity, this example uses a hardcoded list
    neighborhood_options = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
                           'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
                           'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
                           'StoneBr', 'ClearCr', 'NPkVill', 'Blueste']
    neighborhood = st.sidebar.selectbox("Neighborhood", neighborhood_options)

    data = {
        'GrLivArea': gr_liv_area,
        'BedroomAbvGr': bedroom_abv_gr,
        'FullBath': full_bath,
        'YearBuilt': year_built,
        'LotArea': lot_area,
        'Neighborhood': neighborhood
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("Input Features:")
st.write(input_df)

if st.button("Predict House Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.subheader("Predicted House Price:")
        st.success(f"**${prediction:,.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values and ensure the model has been trained correctly.")

st.markdown("---")
st.write("### About the application:")
st.info("""
This application demonstrates data analysis capabilities, Machine Learning model selection (XGBoost),
and result visualization using Streamlit. The model is trained on a house price dataset,
so the accuracy of the prediction depends on the quality and diversity of the training data.
""")

st.write("Made with ❤️ for ML enthusiasts.")