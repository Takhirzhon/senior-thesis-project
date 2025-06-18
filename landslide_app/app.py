# Landslide AI System (v6.14)
import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from PIL import Image

from utils import (
    classify_image,
    predict_geo,
    ensemble,
    predict_from_features,
)

st.set_page_config(page_title="🌍 Landslide AI System", layout="wide")

st.markdown(
    """
    <style>
        .risk-card {
            border-radius: 8px;
            padding: 14px 18px;
            margin:   8px 0 18px 0;
            color: #ffffff;
            font-weight: 600;
            font-size: 1.2rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style='text-align:center;'>🌍 Landslide Susceptibility Prediction</h1>
    <hr>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(
    [
        "🖼️ Image & Coordinate",
        "📝 Manual Feature Input",
    ]
)

with tab1:
    col1, col2 = st.columns(2)

    # ----- Image pipeline -----
    with col1:
        st.subheader("Upload Satellite Image")
        uploaded_img = st.file_uploader(
            "Upload image (.jpg / .png)", type=["jpg", "png"]
        )
        img_prob = None
        if uploaded_img:
            img = Image.open(uploaded_img).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
            if st.button("🔍 Predict from Image"):
                img_prob = classify_image(img)
                st.success(f"✓ Image analysed (p = {img_prob:.2f})")

    with col2:
        st.subheader("Input Coordinates")
        lat = st.number_input("Latitude", format="%.6f")
        lon = st.number_input("Longitude", format="%.6f")
        risk_labels = ["🟢 Very Low", "🟡 Low", "🟠 Moderate", "🔴 High", "🚨 Very High"]
        geo_prob = None
        if (lat or lon) and st.button("🌐 Predict from Coordinates"):
            geo_prob = predict_geo(lat, lon)
            st.success(f"Landslide Risk(p = {geo_prob:.2f}, {risk_labels[int(geo_prob * 4)]})")

with tab2:
    st.subheader("Provide Geo-features Manually")

    feature_inputs = {}
    with st.form("manual_input_form"):
        colA, colB = st.columns(2, gap="large")

        with colA:
            feature_inputs["maxic"]     = st.slider("Max Curvature (maxic)", 0.0, 1.0, 0.27)
            feature_inputs["minic"]     = st.slider("Min Curvature (minic)", 0.0, 1.0, 0.15)
            feature_inputs["slope"]     = st.slider("Slope (°)",             0,   90,  30)
            feature_inputs["elevation"] = st.slider("Elevation (m)",         0, 6000, 1800)
            feature_inputs["flowacc"]   = st.slider("Flow Accumulation",     0,10000, 450)
            feature_inputs["twi"]       = st.slider("TWI",                   0.0, 20.0, 7.5)
            feature_inputs["spi"]       = st.slider("SPI",                   0.0, 20.0, 5.0)

        with colB:
            feature_inputs["planc"]     = st.slider("Plan Curvature",        -1.0, 1.0, 0.02)
            feature_inputs["profc"]     = st.slider("Profile Curvature",     -1.0, 1.0, 0.01)
            feature_inputs["longc"]     = st.slider("Longitudinal Curv.",    -1.0, 1.0, 0.00)
            feature_inputs["rainfall"]  = st.slider("Annual Rainfall (mm)",  0, 3000, 800)
            feature_inputs["ndvi"]      = st.slider("NDVI",                  0.0, 1.0, 0.40)
            feature_inputs["landform"]  = st.number_input("Landform (code)", 0,  20,  4)
            feature_inputs["landcover"] = st.number_input("Land-cover (code)",0,  40, 12)

        submitted = st.form_submit_button("🧮 Predict Risk")

    if submitted:
        prob = predict_from_features(feature_inputs)
        label, colour = (
            ("🟢 Very Low", "#2ecc71") if prob <= 0.10 else
            ("🟡 Low",      "#f1c40f") if prob <= 0.30 else
            ("🟠 Moderate", "#e67e22") if prob <= 0.60 else
            ("🔴 High",     "#e74c3c") if prob <= 0.80 else
            ("🚨 Very High","#8e0000")
        )
        st.markdown(
            f"<div class='risk-card' style='background:{colour};'>"
            f"{label}&nbsp;&nbsp;•&nbsp;&nbsp;Probability&nbsp;<b>{prob:.2f}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )