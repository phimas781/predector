import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App title and description
st.set_page_config(page_title="Gwamz Song Performance Predictor", layout="wide")
st.title("🎵 Gwamz Song Performance Predictor")
st.markdown("""
Predict the expected streams for Gwamz's new songs based on historical data.
Adjust the parameters and click **Predict** to see results.
""")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('gwamz_stream_predictor_v2.pkl')
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Failed to load the prediction model. Please check the logs.")
        st.stop()

# Load model with error handling
try:
    model = load_model()
except Exception as e:
    st.error(f"Critical error loading model: {str(e)}")
    st.stop()

# App title and description
st.set_page_config(page_title="Gwamz Song Performance Predictor", layout="wide")
st.title("🎵 Gwamz Song Performance Predictor")
st.markdown("""
This app predicts the expected streams for Gwamz's new songs based on historical data.
Adjust the parameters below and click **Predict** to see the results.
""")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Calculate days since first release (2021-04-29 based on the data)
first_release = pd.to_datetime('2021-04-29')

# User input function
def user_input_features():
    # Basic info
    album_type = st.sidebar.selectbox('Album Type', ('single', 'album', 'compilation'))
    
    # Release date
    release_date = st.sidebar.date_input("Release Date", datetime(2025, 7, 1))
    release_year = release_date.year
    release_month = release_date.month
    release_dayofweek = release_date.weekday()  # Monday=0, Sunday=6
    days_since_first_release = (release_date - first_release.date()).days
    
    # Track details
    total_tracks_in_album = st.sidebar.slider('Total Tracks in Album', 1, 20, 1)
    available_markets_count = st.sidebar.slider('Available Markets Count', 1, 200, 185)
    track_number = st.sidebar.slider('Track Number', 1, 20, 1)
    disc_number = st.sidebar.slider('Disc Number', 1, 5, 1)
    
    # Content flags
    explicit = st.sidebar.checkbox('Explicit Content', value=True)
    is_sped_up = st.sidebar.checkbox('Sped Up Version', value=False)
    is_remix = st.sidebar.checkbox('Remix/Edit/Club Version', value=False)
    
    # Fixed artist metrics (from data)
    artist_followers = 7937
    artist_popularity = 41
    
    data = {
        'artist_followers': artist_followers,
        'artist_popularity': artist_popularity,
        'album_type': album_type,
        'release_year': release_year,
        'release_month': release_month,
        'release_dayofweek': release_dayofweek,
        'days_since_first_release': days_since_first_release,
        'total_tracks_in_album': total_tracks_in_album,
        'available_markets_count': available_markets_count,
        'track_number': track_number,
        'disc_number': disc_number,
        'explicit': int(explicit),
        'is_sped_up': int(is_sped_up),
        'is_remix': int(is_remix)
    }
    
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display user inputs
st.subheader("Selected Parameters")
st.write(input_df)

# Prediction function
def predict_streams(input_df):
    """Predict streams based on input features"""
    prediction = np.expm1(model.predict(input_df))
    return int(prediction[0])

# Make prediction when button is clicked
if st.sidebar.button('Predict Streams'):
    prediction = predict_streams(input_df)
    
    st.subheader("Prediction Result")
    st.success(f"**Predicted Streams:** {prediction:,}")
    
    # Interpretation
    st.markdown("### Interpretation")
    if prediction > 2000000:
        st.markdown("🔥 **Exceptional Performance Expected!** This track has potential to be one of Gwamz's top performers.")
    elif prediction > 1000000:
        st.markdown("💎 **Strong Performance Expected!** This track should perform well above average.")
    elif prediction > 500000:
        st.markdown("👍 **Good Performance Expected!** This track should perform decently.")
    else:
        st.markdown("💡 **Moderate Performance Expected.** Consider optimizing release strategy or track features.")
    
    # Recommendations based on prediction
    st.markdown("### Recommendations")
    if input_df['is_sped_up'].iloc[0] == 1 and prediction < 500000:
        st.markdown("- Sped up versions in the data have shown mixed performance. Consider releasing a standard version as well.")
    
    if input_df['is_remix'].iloc[0] == 1 and prediction < 300000:
        st.markdown("- Remixes/edits in the data have varied performance. Collaborations might boost streams.")
    
    if input_df['total_tracks_in_album'].iloc[0] > 1 and prediction < 400000:
        st.markdown("- Singles tend to perform better than multi-track releases for this artist.")
    
    if input_df['release_month'].iloc[0] in [11, 12] and prediction < 600000:
        st.markdown("- Holiday season releases might face more competition. Consider promotional activities.")

# Add some analytics
st.sidebar.markdown("---")
st.sidebar.markdown("### Historical Performance Insights")
st.sidebar.markdown("- **Top Performing Tracks**: 'Last Night' (2.9M streams)")
st.sidebar.markdown("- **Best Release Month**: March (multiple high-performing tracks)")
st.sidebar.markdown("- **Sped Up Versions**: Can perform well (e.g., 'Just2 - Sped Up' 2.1M streams)")
st.sidebar.markdown("- **Remixes/Edits**: Performance varies widely (8K to 300K streams)")

# Add footer
st.markdown("---")
st.markdown("""
**Note**: This prediction is based on Gwamz's historical performance data. 
Actual results may vary based on factors not included in the model like marketing efforts, 
current trends, and competition.
""")
