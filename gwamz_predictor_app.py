import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Gwamz Analytics Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('gwamz_streams_predictor_tuned.pkl')
    prophet_model = joblib.load('gwamz_streams_prophet.pkl')
    album_type_encoder = joblib.load('album_type_encoder.pkl')
    version_type_encoder = joblib.load('version_type_encoder.pkl')
    return model, prophet_model, album_type_encoder, version_type_encoder

model, prophet_model, album_type_encoder, version_type_encoder = load_models()

# Preprocess input function
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

# Sidebar inputs
st.sidebar.header("🎛️ Input Parameters")
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

# Main app
st.title("🎵 Gwamz Song Performance Predictor Pro")
st.markdown("Predict future song performance using machine learning and time-series forecasting")

input_data = user_input_features()

col1, col2 = st.columns(2)
with col1:
    if st.button('🔮 Predict Streams', type="primary"):
        processed_input = preprocess_input(input_data)
        prediction = model.predict(processed_input)
        
        st.success(f"🎶 Predicted Streams: **{int(prediction[0]):,}**")
        
        # Performance gauge
        st.subheader("📊 Performance Potential")
        gauge_value = min(max(prediction[0], 0), 3000000)
        st.markdown(f"""
        <div style="background: linear-gradient(to right, #ff5f5f, #f0f2f6, #5fba7d); 
                    height: 30px; border-radius: 5px; position: relative;">
            <div style="position: absolute; left: {gauge_value/3000000*100}%; 
                        top: -10px; width: 2px; height: 50px; background-color: black;">
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
            <span>0</span>
            <span>3M</span>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.subheader("📝 Selected Parameters")
    st.json(input_data)

# Time-series forecast
st.subheader("📈 Future Streams Forecast")
future = prophet_model.make_future_dataframe(periods=12, freq='M')
forecast = prophet_model.predict(future)
fig = px.line(forecast, x='ds', y='yhat', title='12-Month Streams Projection')
fig.update_layout(xaxis_title='Date', yaxis_title='Predicted Streams')
st.plotly_chart(fig, use_container_width=True)

# Feature importance
st.subheader("🔍 Key Performance Factors")
feature_imp = pd.DataFrame({
    'Feature': ['Track Popularity', 'Artist Followers', 'Release Month', 'Version Type', 'Explicit'],
    'Impact': [0.35, 0.25, 0.15, 0.10, 0.05]
})
st.bar_chart(feature_imp.set_index('Feature'))

# Historical data
st.subheader("📋 Historical Performance")
st.dataframe(pd.read_csv('gwamz_data.csv')[['track_name', 'release_date', 'streams', 'track_popularity']]
             .sort_values('streams', ascending=False)
             .head(10))
