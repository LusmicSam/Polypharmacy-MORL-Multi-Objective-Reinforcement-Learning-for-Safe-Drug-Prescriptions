import streamlit as st
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from polypharmacy_env import PolypharmacyEnv

st.set_page_config(page_title="Polypharmacy RL", layout="wide")

st.title("🧠 Polypharmacy Regimen Optimization (Inference Only)")
st.write("Minimal Streamlit interface for your trained GPIPD model.")

MODEL_PATH = "gpipd_model_latest.tar"


# ---------------------------------------------------
# Load agent + environment
# ---------------------------------------------------
@st.cache_resource
def load_agent():
    from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD

    env = PolypharmacyEnv()

    agent = GPIPD(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        net_arch=[128, 128],
        buffer_size=int(5e4),
        initial_epsilon=0.05,
        final_epsilon=0.05,
        log=False,
        dyna=False
    )
    agent.load(MODEL_PATH)
    return agent, env

agent, env = load_agent()
st.success("Model loaded successfully.")


# -----------------------------
# Weight Input
# -----------------------------
st.sidebar.header("Weights (Efficacy / DDI / Tolerability)")

w1 = st.sidebar.slider("Efficacy Weight", 0.0, 1.0, 0.3)
w2 = st.sidebar.slider("DDI Weight", 0.0, 1.0, 0.3)
w3 = st.sidebar.slider("Tolerability Weight", 0.0, 1.0, 0.4)

weight_vec = np.array([w1, w2, w3], dtype=np.float32)
weight_vec /= weight_vec.sum()

st.sidebar.write("Normalized:", weight_vec.tolist())


# -----------------------------
# Run 1-step simulation
# -----------------------------
st.header("🔍 Single-Step Simulation")

obs, _ = env.reset()
st.write("Initial Observation:", obs)

if st.button("Run Step"):
    action = agent.eval(obs, weight_vec)
    next_obs, reward, done, trunc, info = env.step(action)

    st.write("### Action:", action)
    st.write("Reward:", reward)
    st.write("Next Obs:", next_obs)
    st.write("Info:", info)
    st.success("Inference completed.")


# -----------------------------
# Optional scatter visualization
# -----------------------------
st.header("📊 Upload Evaluation CSV (Optional)")

uploaded = st.file_uploader("Upload phase7 or phase8 CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["efficacy"], df["neg_ddi"], c=df["neg_tol"], cmap="viridis")
    ax.set_xlabel("Efficacy")
    ax.set_ylabel("DDI Risk")
    ax.set_title("Pareto Scatter")
    st.pyplot(fig)
