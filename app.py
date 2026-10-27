
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -------------------- CONFIG --------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
LOGO_PATH = BASE_DIR / "assets" / "logo.svg"

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
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #172033;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2941, #123e50);
}
[data-testid="stSidebar"] * {
    color: #f1f7fb;
}
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 18px;
}
.brand img {
    width: 58px;
    height: 58px;
}
.brand-name {
    font-family: 'Manrope', sans-serif;
    font-size: 23px;
    font-weight: 800;
    color: white;
}
.brand-name span {
    color: #68e0b5;
}
.brand-caption {
    font-size: 10px;
    letter-spacing: 1.4px;
    color: #c6dbe5;
}
.side-item {
    padding: 11px 13px;
    margin: 7px 0;
    border-radius: 10px;
    background: rgba(255,255,255,0.07);
    font-weight: 600;
}
.side-note {
    color: #c6dbe5;
    font-size: 12px;
    line-height: 1.7;
}
.hero {
    background: linear-gradient(115deg, #0d2941, #235b70);
    color: white;
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 25px;
    border: 1px solid #315a70;
}
.hero h1 {
    color: white !important;
    font-size: 36px;
    margin: 10px 0;
}
.hero p {
    color: #dbeafe;
    font-size: 15px;
    margin-bottom: 0;
}
.eyebrow {
    color: #8ce2ce;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
}
.panel {
    background: white;
    padding: 23px;
    border: 1px solid #e3eaf0;
    border-radius: 17px;
    margin-bottom: 18px;
}
.metric-card {
    background: #f8fbfd;
    border: 1px solid #e4edf2;
    padding: 15px;
    border-radius: 13px;
    min-height: 85px;
}
.metric-label {
    color: #64748b;
    font-size: 12px;
    margin-bottom: 6px;
}
.metric-value {
    color: #172033;
    font-family: 'Manrope', sans-serif;
    font-size: 21px;
    font-weight: 800;
}
.result-card {
    background: linear-gradient(135deg, #e0f7ef, #f1fcf7);
    border: 1px solid #b9ead7;
    border-radius: 17px;
    padding: 23px;
    margin-top: 18px;
}
.result-label {
    color: #28745c;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 1px;
}
.result-price {
    color: #105b46;
    font-family: 'Manrope', sans-serif;
    font-size: 36px;
    font-weight: 800;
    margin: 8px 0;
}
div.stButton > button {
    background: #167d68;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 47px;
    font-weight: 700;
    width: 100%;
}
div.stButton > button:hover {
    background: #106450;
    color: white;
    border: none;
}
div[data-testid="stNumberInput"] input {
    border-radius: 9px;
}
.small-note {
    color: #64748b;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python train_model.py")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(MODEL_PATH))

# -------------------- SIDEBAR BRANDING --------------------
with st.sidebar:
    if LOGO_PATH.exists():
        logo_svg = LOGO_PATH.read_text(encoding="utf-8")
        import base64
        logo_base64 = base64.b64encode(
            logo_svg.encode("utf-8")
        ).decode("utf-8")
        logo_src = f"data:image/svg+xml;base64,{logo_base64}"
        st.markdown(f"""
        <div class="brand">
            <img src="{logo_src}" alt="HomeValue logo">
            <div>
                <div class="brand-name">Home<span>Value</span></div>
                <div class="brand-caption">AI PROPERTY ESTIMATOR</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("## 🏡 HomeValue")
        st.caption("AI PROPERTY ESTIMATOR")
        st.warning("Logo missing: add assets/logo.svg")

    st.divider()
    st.markdown('<div class="side-item">⌂ &nbsp; Home</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-item">▥ &nbsp; About Model</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-item">ⓘ &nbsp; About Project</div>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown("**PROJECT DETAILS**")
    st.markdown(
        '<div class="side-note">'
        '⚙️ &nbsp; Model<br>Random Forest Regressor<br><br>'
        '▤ &nbsp; Dataset<br>Ames, Iowa, USA<br><br>'
        '✦ &nbsp; Interface<br>Streamlit'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<div class="side-note">Better data.<br>'
        'Smarter predictions.<br>Brighter futures.</div>',
        unsafe_allow_html=True,
    )

# -------------------- HERO HEADER --------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ SMART PROPERTY INSIGHTS</div>
    <h1>Find your home's estimated value.</h1>
    <p>Enter property details and let your trained machine learning
    model estimate a price based on historical housing data.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- PROPERTY INPUTS --------------------
st.markdown("## Property details")
st.caption("Adjust the values below, then calculate your estimate.")

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

    predict_clicked = st.button("✦  Predict House Price")

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 📋 Property summary")

    summary_items = [
        ("Living area", f"{area:,} sq ft"),
        ("Overall quality", f"{quality}/10"),
        ("Garage capacity", f"{garage} cars"),
        ("Year built", str(year)),
        ("Basement area", f"{basement:,} sq ft"),
    ]

    for label, value in summary_items:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")

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
            <div class="result-label">ESTIMATED HOUSE PRICE</div>
            <div class="result-price">${prediction:,.0f}</div>
            <div style="color:#28745c;">
                ✓ Prediction generated successfully by your ML model.
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
