
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------- CONFIG --------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"

st.set_page_config(
    page_title="HomeValue | Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #f5f7fb;
    color: #172033;
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #172033;
}
.hero {
    background: linear-gradient(120deg, #142d4e, #235b70);
    color: white;
    padding: 34px;
    border-radius: 22px;
    margin-bottom: 25px;
}
.hero h1 {
    color: white !important;
    font-size: 38px;
    margin: 5px 0 10px 0;
}
.hero p {
    color: #dbeafe;
    font-size: 16px;
    margin-bottom: 0;
}
.eyebrow {
    color: #8ce2ce;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 700;
}
.panel {
    background: white;
    padding: 25px;
    border: 1px solid #e6eaf0;
    border-radius: 18px;
    margin-bottom: 18px;
}
.metric-card {
    background: white;
    border: 1px solid #e6eaf0;
    padding: 20px;
    border-radius: 16px;
    min-height: 105px;
}
.metric-label {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 8px;
}
.metric-value {
    color: #172033;
    font-family: 'Manrope', sans-serif;
    font-size: 23px;
    font-weight: 800;
}
.result-card {
    background: linear-gradient(135deg, #e0f7ef, #effcf7);
    border: 1px solid #b9ead7;
    border-radius: 18px;
    padding: 26px;
    margin-top: 18px;
}
.result-label {
    color: #28745c;
    font-weight: 600;
    font-size: 14px;
}
.result-price {
    color: #105b46;
    font-family: 'Manrope', sans-serif;
    font-size: 38px;
    font-weight: 800;
    margin: 8px 0;
}
div.stButton > button {
    background: #167d68;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
    width: 100%;
}
div.stButton > button:hover {
    background: #106450;
    color: white;
    border: none;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e6eaf0;
}
.small-note {
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🏡 HomeValue")
    st.caption("AI-powered house price demo")
    st.divider()
    st.markdown("### About this project")
    st.write(
        "Estimate a house price from selected property features "
        "using a trained Random Forest model."
    )
    st.divider()
    st.markdown("**Model:** Random Forest Regressor")
    st.markdown("**Dataset:** Ames, Iowa, USA")
    st.markdown("**Interface:** Streamlit")
    st.divider()
    st.caption("Educational project • Not a real estate valuation tool")

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">SMART PROPERTY INSIGHTS</div>
    <h1>Find your home's estimated value.</h1>
    <p>Enter property details and let your trained machine learning
    model estimate a price based on historical housing data.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python train_model.py")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(MODEL_PATH))

# -------------------- INPUTS --------------------
st.markdown("## Property details")
st.write("Adjust the values below, then calculate your estimate.")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 🏠 Home characteristics")

    quality = st.slider(
        "Overall quality",
        min_value=1,
        max_value=10,
        value=7,
        help="Overall material and finish quality, rated from 1 to 10.",
    )

    area = st.number_input(
        "Living area (square feet)",
        min_value=100,
        max_value=10000,
        value=2000,
        step=50,
    )

    basement = st.number_input(
        "Basement area (square feet)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50,
    )

    c1, c2 = st.columns(2)
    with c1:
        garage = st.number_input(
            "Garage capacity (cars)",
            min_value=0,
            max_value=5,
            value=2,
        )
    with c2:
        year = st.number_input(
            "Year built",
            min_value=1800,
            max_value=2026,
            value=2005,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button(
        "✨  Predict House Price",
        use_container_width=True,
    )

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📋 Property summary")
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">'
        f'Living area</div><div class="metric-value">{area:,} sq ft'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">'
        f'Overall quality</div><div class="metric-value">'
        f'{quality}/10</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">'
        f'Garage capacity</div><div class="metric-value">'
        f'{garage} cars</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">'
        f'Year built</div><div class="metric-value">'
        f'{year}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
if predict_clicked:
    input_data = pd.DataFrame([{
        "OverallQual": quality,
        "GrLivArea": area,
        "GarageCars": garage,
        "TotalBsmtSF": basement,
        "YearBuilt": year,
    }])

    try:
        prediction = float(model.predict(input_data)[0])

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">YOUR ESTIMATED HOUSE PRICE</div>
            <div class="result-price">${prediction:,.0f}</div>
            <div style="color:#28745c;">
                Prediction generated successfully by your ML model.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Prediction inputs")
        st.dataframe(input_data, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

else:
    st.info("Enter your property details and click Predict House Price.")

# -------------------- FOOTER --------------------
st.divider()
st.markdown(
    '<p class="small-note">Disclaimer: This project uses the Ames, '
    'Iowa housing dataset. Prices are model estimates in dataset currency '
    'and should not be used as actual property valuations in Nepal.</p>',
    unsafe_allow_html=True,
)
