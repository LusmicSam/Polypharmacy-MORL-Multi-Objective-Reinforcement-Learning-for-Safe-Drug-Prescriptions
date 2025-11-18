import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px
import json
import os

# ================================
# CONFIG
# ================================
MODEL_CLEAN_PATH = "gpipd_model_clean.pt"
PHASE8_DIR = "phase8_results"

st.set_page_config(
    page_title="Polypharmacy MORL Explorer",
    layout="wide"
)

st.title("💊 Polypharmacy MORL — Interactive Explorer (Phase 7 & 8 Results)")


# ================================
# Q-NETWORK ARCHITECTURE
# (matches your saved shapes)
# ================================
class CleanQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights_features = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
        )
        self.state_features = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6)   # 2 actions × 3 reward dims
        )

    def forward(self, state, weight):
        w = self.weights_features(weight)
        s = self.state_features(state)
        fused = w * s
        return self.net(fused)


# ================================
# LOAD CLEAN MODEL
# ================================
@st.cache_resource
def load_clean_model():
    if not os.path.exists(MODEL_CLEAN_PATH):
        raise FileNotFoundError(f"{MODEL_CLEAN_PATH} not found!")

    ckpt = torch.load(MODEL_CLEAN_PATH, map_location="cpu")

    q_nets = []
    for q_state in ckpt["q_nets"]:
        q = CleanQNet()
        q.load_state_dict(q_state)
        q.eval()
        q_nets.append(q)

    weight_support = ckpt["weight_support"]
    return q_nets, weight_support


st.header("Model Loading")

try:
    q_networks, weight_support = load_clean_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()


# ================================
# ACTION MAP
# ================================
ACTION_MAP = {
    0: "Reduce-risk policy (deprescribe candidate)",
    1: "Maintain regimen (conservative)"
}


# ================================
# PHASE 8 DATA LOADING
# ================================
def load_phase8():
    files = {
        "flat": os.path.join(PHASE8_DIR, "phase8_flat_results.csv"),
        "pareto": os.path.join(PHASE8_DIR, "pareto_front_mean_by_weight.csv"),
        "raw": os.path.join(PHASE8_DIR, "phase8_raw_results.json")
    }

    dfs = {}
    for k, path in files.items():
        if path.endswith(".csv") and os.path.exists(path):
            dfs[k] = pd.read_csv(path)
        elif path.endswith(".json") and os.path.exists(path):
            with open(path, "r") as f:
                dfs[k] = json.load(f)

    return dfs


phase8 = load_phase8()
st.success("Phase 8 results loaded.")


# ================================
# UI — VISUALIZATION SELECTOR
# ================================
st.header("📊 Phase 8 Visual Explorer")

viz_type = st.selectbox("Choose visualization", [
    "Pareto Front",
    "Action Frequency Heatmap",
    "Reward Correlation",
    "Radar Tradeoff Plots",
    "Raw CSV Viewer"
])

# ================================
# PARETO FRONT
# ================================
if viz_type == "Pareto Front":
    pareto_df = phase8["pareto"]

    st.subheader("Pareto Front — Mean Rewards by Weight")

    x_axis = st.selectbox("X axis", pareto_df.columns, index=1)
    y_axis = st.selectbox("Y axis", pareto_df.columns, index=2)
    color = st.selectbox("Color", pareto_df.columns, index=3)

    fig = px.scatter(
        pareto_df,
        x=x_axis,
        y=y_axis,
        color=color,
        hover_data=list(pareto_df.columns)
    )
    st.plotly_chart(fig, use_container_width=True)


# ================================
# ACTION FREQUENCY HEATMAP
# ================================
elif viz_type == "Action Frequency Heatmap":
    img_path = os.path.join(PHASE8_DIR, "action_frequency_heatmap.png")
    st.image(img_path, caption="Action Heatmap", use_column_width=True)


# ================================
# REWARD CORRELATION MATRIX
# ================================
elif viz_type == "Reward Correlation":
    img_path = os.path.join(PHASE8_DIR, "reward_correlation_matrix.png")
    st.image(img_path, caption="Reward Correlation", use_column_width=True)


# ================================
# RADAR TRADEOFF PLOTS
# ================================
elif viz_type == "Radar Tradeoff Plots":
    img_path = os.path.join(PHASE8_DIR, "tradeoff_radar_plots.png")
    st.image(img_path, caption="Radar Plots", use_column_width=True)


# ================================
# RAW CSV VIEWER
# ================================
elif viz_type == "Raw CSV Viewer":
    csv_choice = st.selectbox("Choose CSV", list(phase8.keys()))
    if csv_choice in phase8:
        st.dataframe(phase8[csv_choice])


# ================================
# Simple Inference Demo
# ================================
st.header("🔮 Try Inference (Toy Demo)")

state = st.text_input("Enter state vector (4 floats)", "0.1, 0.2, 0.05, 0.7")
weight = st.text_input("Enter weight vector (3 floats)", "0.4, 0.3, 0.3")

if st.button("Predict Action"):
    try:
        s = torch.tensor([float(x) for x in state.split(",")], dtype=torch.float32)
        w = torch.tensor([float(x) for x in weight.split(",")], dtype=torch.float32)

        outputs = q_networks[0](s, w).detach().numpy()
        psi_a0 = outputs[:3]
        psi_a1 = outputs[3:]

        chosen = 0 if psi_a0.sum() > psi_a1.sum() else 1

        st.write(f"**Action 0 ψ = {psi_a0}**")
        st.write(f"**Action 1 ψ = {psi_a1}**")
        st.success(f"Chosen Action: {ACTION_MAP[chosen]}")

    except:
        st.error("Invalid input format. Use comma-separated floats.")

