import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
from sklearn import __version__ as sklearn_version

# Configure page
st.set_page_config(
    page_title="Gwamz Song Predictor PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom loader with version check
@st.cache_resource
def load_model():
    try:
        # Version compatibility check
        st.sidebar.markdown(f"""
        **Environment Info:**
        - Python: {sys.version.split()[0]}
        - scikit-learn: {sklearn_version}
        """)
        
        # Load model with error handling
        model = joblib.load('gwamz_stream_predictor_final.pkl')
        return model
    except Exception as e:
        st.error(f"""
        ## Model Loading Failed
        **Error:** {str(e)}
        
        Please ensure:
        1. The model file exists in the same directory
        2. Package versions match requirements.txt
        """)
        st.stop()

model = load_model()

# --- UI Components ---
st.title("🎵 Gwamz Song Performance Predictor PRO")
st.markdown("""
Predict the streaming performance of new Gwamz tracks based on historical data patterns.
""")

with st.sidebar:
    st.header("Configuration")
    st.info("Adjust these parameters to match your new release")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Release Details")
        release_date = st.date_input(
            "Release Date",
            datetime(2025, 7, 1),
            help="Strategic release dates can impact performance"
        )
        album_type = st.selectbox(
            "Album Type",
            ["single", "album", "EP"],
            index=0,
            help="Singles typically perform better for this artist"
        )
        total_tracks = st.slider(
            "Total Tracks in Release",
            1, 10, 1,
            help="More tracks may dilute streaming numbers"
        )
        
    with col2:
        st.subheader("Track Features")
        is_sped_up = st.checkbox(
            "Sped Up Version",
            help="Sped-up versions have shown variable performance"
        )
        is_remix = st.checkbox(
            "Remix/Edit Version",
            help="Remixes may attract different audiences"
        )
        explicit = st.checkbox(
            "Explicit Content",
            value=True,
            help="Explicit tracks tend to perform slightly better"
        )
    
    submitted = st.form_submit_button("Predict Streams", type="primary")

# Prediction Logic
if submitted:
    # Prepare input data (aligned with training features)
    input_data = {
        'artist_followers': 7937,  # From historical data
        'artist_popularity': 41,   # From historical data
        'album_type': album_type,
        'release_year': release_date.year,
        'release_month': release_date.month,
        'days_since_first_release': (release_date - pd.to_datetime('2021-04-29').date()).days,
        'total_tracks_in_album': total_tracks,
        'explicit': int(explicit),
        'is_sped_up': int(is_sped_up),
        'is_remix': int(is_remix)
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    try:
        # Make prediction
        log_pred = model.predict(input_df)
        prediction = int(np.expm1(log_pred)[0])
        
        # Display results
        st.success(f"### Predicted Streams: {prediction:,}")
        
        # Performance interpretation
        if prediction > 1500000:
            st.balloons()
            st.markdown("""
            🔥 **Exceptional Potential**  
            This configuration matches Gwamz's top-performing tracks!
            """)
        elif prediction > 800000:
            st.markdown("""
            💎 **Strong Performance Expected**  
            Likely to perform better than average releases.
            """)
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("""
Note: Predictions are based on historical patterns and don't account for marketing efforts or current trends.
""")
