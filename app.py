import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="California Housing Predictor")

# THE STYLING

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #00d4ff;
        font-family: 'Courier New', monospace;
    }
    .stApp {
        background-image: url('https://i.postimg.cc/BvVB3L33/jose-rago-LNl-J0WZHi-Es-unsplash.jpg'); 
        background-size: 100% 100%;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 50px;
        box-shadow: 0 0 20px #00d4ff;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.1);
        box-shadow: 0 0 30px #ff6b6b;
    }

    /* Increase font size of ALL slider labels */
.stSlider label {
    font-size: 50px ;
    font-weight: bold;
    color: white; 
}



    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with app info and instructions
st.sidebar.title("About This App")
st.sidebar.info(
    """
    This app predicts California housing prices based on key features.
    Use the sliders to adjust feature values, then click Predict.

    The model loaded from `model.pkl` and scaler from `scaler.pkl` are used.
    


    """
)

st.title("Welcome To California Housing Price Prediction App")


# LOAD MODEL

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        st.success("Data Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load Data Model: {str(e)}")
else:
    st.warning("Model or scaler file not found. Using mock prediction for now.")



# FOR INPUTING THE FEATURE VALUES

st.subheader("Input Feature Values")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("Median Income (MedInc)   1 -> $10,000", 0.0, 150.0, 3.5, help="Median income in block group (in tens of thousands of USD)")
    HouseAge = st.slider("House Age", 1, 1000, 20, help="Median house age in block group years")
    AveRooms = st.slider("Average Rooms", 0.0, 100.0, 5.0, help="Average number of rooms per household")
    AveBedrms = st.slider("Average Bedrooms", 0.0, 100.0, 1.0, help="Average number of bedrooms per household")
    
with col2:
    Population = st.slider("Population", 1, 50000, 1000, help="Block group population")
    AveOccup = st.slider("Average Occupancy", 0.0, 100.0, 3.0, help="Average number of occupants per household")
    Latitude = st.slider("Latitude", 32.0, 42.0, 34.0, help="Block group latitude")
    Longitude = st.slider("Longitude", -125.0, -114.0, -118.0, help="Block group longitude")



# FOR PREDICTING

if st.button("Predict"):
    with st.spinner("Predicting..."):
        X = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

        try:
            if model and scaler:
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]
                price = pred * 100000
                st.success(f"Predicted Price: ${price:,.0f} USD")

                

                # TO KNOW THE IMPACT OF EACH FEATURE ON THE PREDICTION
                
                st.subheader("Prediction Explanation")
                feature_names = ["Median Income", "House Age", "Average Rooms", "Average Bedrooms", 
                                 "Population", "Average Occupancy", "Latitude", "Longitude"]
                if hasattr(model, "coef_"):
                    coefs = model.coef_
                    importance = coefs * X_scaled[0]
                else:
                    importance = np.array([0.4, 0.01, 0.05, 0.02, 0.005, 0.01, 0.001, 0.001]) * X_scaled[0]

                # TO EXPLAIN THE PREDICTIONS

                explanation = {name: float(f"{val:.3f}") for name, val in zip(feature_names, importance)}
                for feat, val in explanation.items():
                    st.write(f"{feat}: {val}")


                # ADD AN EXPANDER TO SHOW THE BAR CHART FOR DETAILED EXPLANATION
                with st.expander("Show Prediction Explanation Chart"):
                    st.bar_chart(explanation)

               

               
            # IF THERE IS NO MODEL

            else:
                # Mock prediction
                mock_pred = (MedInc * 0.4) + (HouseAge * 0.01)
                mock_price = mock_pred * 100000
                st.info(f"(Mock) Predicted Price: {mock_price:,.0f} USD")

                st.subheader("Prediction Explanation (Mock)")
                st.write(f"Median Income contribution: {MedInc * 0.4:.3f}")
                st.write(f"House Age contribution: {HouseAge * 0.01:.3f}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
