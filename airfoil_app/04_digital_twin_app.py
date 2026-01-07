import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import aerosandbox as asb

# --- CONFIGURATION ---
st.set_page_config(page_title="AI-CFD Digital Twin", layout="wide")

# --- LOAD RESOURCES (Cached for speed) ---
@st.cache_resource
def load_brain():
    model = tf.keras.models.load_model('airfoil_ai_model.keras')
    scaler_X = joblib.load('scaler_X.pkl')
    # We don't strictly need scaler_y for prediction if we inverse transform manually, 
    # but let's assume raw output for now or load if you saved it.
    # based on your previous logs, you saved 'scaler_X.pkl'.
    return model, scaler_X

try:
    model, scaler_X = load_brain()
    st.success("✅ AI Neural Network Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- SIDEBAR: FLIGHT CONTROLS ---
st.sidebar.header("✈️ Flight Conditions")
alpha = st.sidebar.slider("Angle of Attack (α)", -10.0, 15.0, 0.0, 0.1)
reynolds = st.sidebar.slider("Reynolds Number (Re)", 500000, 2000000, 1000000, 50000)

st.sidebar.header("📐 Wing Geometry (NACA)")
camber = st.sidebar.slider("Max Camber (%)", 0.0, 6.0, 2.0, 0.1) / 100.0
camber_loc = st.sidebar.slider("Camber Location (10th)", 0.0, 9.0, 4.0, 1.0) / 10.0
thickness = st.sidebar.slider("Thickness (%)", 8.0, 24.0, 12.0, 1.0) / 100.0

# --- MAIN DASHBOARD ---
st.title("🌪️ Real-Time AI Aerodynamics Twin")
st.markdown("Use the controls on the left to modify the airfoil. The AI predicts performance in **milliseconds**.")

# 1. VISUALIZE THE SHAPE
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Geometry Visualization")
    
    # Generate coordinates for plotting using Aerosandbox
    # We reconstruct the NACA string roughly from inputs
    d1 = int(camber * 100)
    d2 = int(camber_loc * 10)
    d34 = int(thickness * 100)
    naca_name = f"naca{d1}{d2}{d34:02d}"
    
    try:
        airfoil = asb.Airfoil(naca_name)
        coords = airfoil.coordinates
        
        fig_shape, ax_shape = plt.subplots(figsize=(10, 3))
        ax_shape.plot(coords[:,0], coords[:,1], 'k-', linewidth=2)
        ax_shape.fill(coords[:,0], coords[:,1], 'gray', alpha=0.3)
        ax_shape.set_aspect('equal')
        ax_shape.set_xlim(-0.05, 1.05)
        ax_shape.set_ylim(-0.2, 0.2)
        ax_shape.grid(True, alpha=0.3)
        ax_shape.set_title(f"Shape: {naca_name} (Generated)")
        st.pyplot(fig_shape)
        
    except:
        st.warning("Invalid NACA parameters for visualization")

# 2. RUN AI INFERENCE
# Prepare Input Vector: ['camber', 'camber_loc', 'thickness', 'alpha', 'Re']
input_data = np.array([[camber, camber_loc, thickness, alpha, reynolds]])

# Scale the input
input_scaled = scaler_X.transform(input_data)

# Predict
prediction_scaled = model.predict(input_scaled, verbose=0)
# Note: If we scaled Y during training, we need to unscale it here. 
# Looking at your previous code, you scaled X but did you scale Y? 
# Your logs said "Scalers saved: scaler_X.pkl, scaler_y.pkl". 
# IF you scaled Y, we need to load scaler_y and inverse_transform.

# Let's try to load scaler_y to be safe
try:
    scaler_y = joblib.load('scaler_y.pkl')
    prediction = scaler_y.inverse_transform(prediction_scaled)
    cl_pred = prediction[0][0]
    cd_pred = prediction[0][1]
except:
    # If Y wasn't scaled or file missing, assume raw output
    cl_pred = prediction_scaled[0][0]
    cd_pred = prediction_scaled[0][1]


# 3. DISPLAY RESULTS
with col2:
    st.subheader("AI Prediction")
    
    # Calculate Efficiency
    ld_ratio = cl_pred / cd_pred if cd_pred > 0 else 0
    
    st.metric(label="Lift Coefficient (CL)", value=f"{cl_pred:.4f}")
    st.metric(label="Drag Coefficient (CD)", value=f"{cd_pred:.4f}")
    st.metric(label="Efficiency (L/D)", value=f"{ld_ratio:.1f}")
    
    if cl_pred > 1.2:
        st.warning("⚠️ High Lift - Approaching Stall Risk")
    if cd_pred < 0.01:
        st.success("✅ Laminar Flow Region")

# 4. LIVE POLAR PLOT (Context)
st.markdown("---")
st.subheader("Flight Envelope Context")
st.info("The red dot shows your current operating point relative to a standard drag polar.")

# Fake background polar for context
alpha_range = np.linspace(-10, 15, 50)
cl_ideal = 2 * np.pi * np.radians(alpha_range)
cd_ideal = 0.01 + 0.05 * cl_ideal**2 

fig_polar, ax_polar = plt.subplots(1, 2, figsize=(12, 4))

# CL vs Alpha
ax_polar[0].plot(alpha_range, cl_ideal, 'b--', alpha=0.3, label="Ideal Theory")
ax_polar[0].plot(alpha, cl_pred, 'ro', markersize=10, label="AI Prediction")
ax_polar[0].set_xlabel("Alpha (deg)")
ax_polar[0].set_ylabel("CL")
ax_polar[0].grid(True)
ax_polar[0].legend()

# Drag Polar
ax_polar[1].plot(cd_ideal, cl_ideal, 'b--', alpha=0.3, label="Ideal Theory")
ax_polar[1].plot(cd_pred, cl_pred, 'ro', markersize=10, label="AI Prediction")
ax_polar[1].set_xlabel("CD")
ax_polar[1].set_ylabel("CL")
ax_polar[1].grid(True)

st.pyplot(fig_polar)