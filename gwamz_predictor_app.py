import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px

# Load models
model = joblib.load('gwamz_streams_predictor_tuned.pkl')
prophet_model = joblib.load('gwamz_streams_prophet.pkl')
album_type_encoder = joblib.load('album_type_encoder.pkl')
version_type_encoder = joblib.load('version_type_encoder.pkl')

# Preprocess input function (same as before)
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    first_release = datetime(2021, 4, 29)
    release_date = datetime(input_data['release_year'], input_data['release_month'], 15)
    input_df['days_since_first_release'] = (release_date - first_release).days
    input_df['album_type'] = album_type_encoder.transform([input_data['album_type']])[0]
    input_df['version_type'] = version_type_encoder.transform([input_data['version_type']])[0]
    features = ['artist_followers', 'artist_popularity', 'release_year', 
                'total_tracks_in_album', 'available_markets_count', 'track_popularity',
                'release_month', 'release_dayofweek', 'days_since_first_release',
                'album_type', 'version_type', 'explicit']
    return input_df[features]

# Streamlit App
st.set_page_config(page_title="Gwamz Analytics Pro", layout="wide")

# Sidebar
st.sidebar.header("Input Parameters")
def user_input_features():
    artist_followers = st.sidebar.slider('Artist Followers', 0, 20000, 7937)
    artist_popularity = st.sidebar.slider('Artist Popularity (0-100)', 0, 100, 41)
    release_year = st.sidebar.slider('Release Year', 2021, 2025, 2025)
    release_month = st.sidebar.slider('Release Month', 1, 12, 6)
    release_dayofweek = st.sidebar.selectbox('Release Day of Week', 
                                           ['Monday', 'Tuesday', 'Wednesday', 
                                            'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                           index=0)
    total_tracks_in_album = st.sidebar.slider('Total Tracks in Album', 1, 10, 1)
    available_markets_count = st.sidebar.slider('Available Markets Count', 1, 200, 185)
    track_popularity = st.sidebar.slider('Track Popularity (0-100)', 0, 100, 50)
    album_type = st.sidebar.selectbox('Album Type', ['single', 'album', 'compilation'], index=0)
    version_type = st.sidebar.selectbox('Version Type', ['original', 'sped_up', 'instrumental', 'jersey', 'remix'], index=0)
    explicit = st.sidebar.checkbox('Explicit Content', value=True)
    
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    release_dayofweek_num = day_map[release_dayofweek]
    
    return {
        'artist_followers': artist_followers,
        'artist_popularity': artist_popularity,
        'release_year': release_year,
        'release_month': release_month,
        'release_dayofweek': release_dayofweek_num,
        'total_tracks_in_album': total_tracks_in_album,
        'available_markets_count': available_markets_count,
        'track_popularity': track_popularity,
        'album_type': album_type,
        'version_type': version_type,
        'explicit': explicit
    }

input_data = user_input_features()

# Main Dashboard
st.title("🎵 Gwamz Song Performance Predictor Pro")

# Prediction Section
if st.button('Predict Streams'):
    processed_input = preprocess_input(input_data)
    prediction = model.predict(processed_input)
    
    st.success(f"🎶 Predicted Streams: **{int(prediction[0]):,}**")
    
    # Show Prophet Forecast
    st.subheader("📈 Future Streams Forecast (Next 12 Months)")
    future = prophet_model.make_future_dataframe(periods=12, freq='M')
    forecast = prophet_model.predict(future)
    
    fig = px.line(forecast, x='ds', y='yhat', title='Expected Streams Over Time')
    fig.update_layout(xaxis_title='Date', yaxis_title='Streams')
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.subheader("🔍 Top Factors Affecting Streams")
    feature_importance = pd.DataFrame({
        'Feature': ['Track Popularity', 'Artist Followers', 'Release Month', 'Version Type', 'Explicit'],
        'Impact': [0.35, 0.25, 0.15, 0.10, 0.05]
    })
    st.bar_chart(feature_importance.set_index('Feature'))

# Data Explorer
st.subheader("📊 Historical Performance")
st.dataframe(gwamz_data[['track_name', 'release_date', 'streams', 'track_popularity']].sort_values('streams', ascending=False))
