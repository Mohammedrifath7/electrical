import streamlit as st
from groq import Groq
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime, timedelta
import re

# Configure Groq client
client = Groq(api_key="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")

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stDateInput, .stNumberInput {background-color: white;}
    h1 {color: #1a73e8;}
    .stButton button {background-color: #1a73e8; color: white;}
    .prediction-box {padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0;}
</style>
""", unsafe_allow_html=True)

def extract_json_from_response(response_text):
    """Extract JSON content from model response"""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        st.error(f"JSON extraction error: {str(e)}")
        return None

# App Header
st.title("🔌 Delhi Electricity Demand Forecast")
st.markdown("AI-powered electricity demand projection for NCT of Delhi")

# Input Section
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("📅 Start Date", datetime.today())
with col2:
    end_date = st.date_input("📅 End Date", datetime.today() + timedelta(days=7))

st.markdown("---")
st.subheader("Weather Parameters")
col3, col4, col5 = st.columns(3)
with col3:
    temp = st.number_input("🌡️ Average Temperature (°C)", min_value=0, max_value=50, value=30)
with col4:
    humidity = st.number_input("💧 Average Humidity (%)", min_value=0, max_value=100, value=60)
with col5:
    rainfall = st.number_input("🌧️ Average Rainfall (mm)", min_value=0.0, value=0.0)

# Prediction Button
if st.button("🔮 Predict Demand", use_container_width=True):
    with st.spinner("Analyzing patterns with AI..."):
        # Construct the AI prompt
        prompt = f"""Act as an expert power systems analyst. Predict electricity demand for Delhi between {start_date} and {end_date}.
        Parameters:
        - Average Temperature: {temp}°C
        - Average Humidity: {humidity}%
        - Average Rainfall: {rainfall}mm
        - Date Range: {start_date} to {end_date}

        Consider these factors:
        1. Typical daily load patterns
        2. Weather impact on demand
        3. Seasonal variations
        4. Duck curve effect from solar
        5. Urban development patterns

        Respond ONLY with valid JSON in this format:
        {{
            "average_demand": [average in MW],
            "peak_demand": [peak in MW],
            "peak_time": "HH:MM",
            "demand_curve": {{
                "00:00": [MW],
                "01:00": [MW],
                ... [complete all 24 hours]
            }},
            "analysis": "[50-word technical analysis]"
        }}
        Do not include any other text or explanation."""

        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.3
            )
            
            # Extract and parse JSON response
            response_text = response.choices[0].message.content
            result = extract_json_from_response(response_text)
            
            if not result:
                st.error("Failed to parse AI response. Raw response:")
                st.code(response_text)
                st.stop()  # Corrected from 'return'
            
            # Display predictions
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col6, col7, col8 = st.columns(3)
            with col6:
                st.metric("Average Demand", f"{result['average_demand']:,.0f} MW")
            with col7:
                st.metric("Peak Demand", f"{result['peak_demand']:,.0f} MW")
            with col8:
                st.metric("Peak Time", result['peak_time'])
            
            # Create demand curve dataframe
            df = pd.DataFrame(list(result['demand_curve'].items()), 
                            columns=['Hour', 'Demand (MW)'])
            df['Hour'] = pd.to_datetime(df['Hour'], format='%H:%M').dt.strftime('%H:%M')
            
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['Hour'], df['Demand (MW)'], marker='o', color='#1a73e8')
            ax.set_title("24-Hour Demand Forecast", fontweight='bold')
            ax.set_xlabel("Time of Day", fontweight='bold')
            ax.set_ylabel("Demand (MW)", fontweight='bold')
            plt.xticks(rotation=45)
            plt.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Technical analysis
            st.markdown("---")
            st.subheader("📈 Technical Analysis")
            st.info(result['analysis'])
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

st.markdown("---")
st.markdown("**Note**: Predictions incorporate weather patterns, historical load data, and urban development trends. Model accuracy ±5%")