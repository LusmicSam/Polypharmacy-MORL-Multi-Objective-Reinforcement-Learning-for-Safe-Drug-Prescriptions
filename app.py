import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
import json
import os
from polypharmacy_env import PolypharmacyEnv
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD

st.set_page_config(layout="wide", page_title="Polypharmacy RL Analysis")

st.title("💊 Multi-Objective RL for Polypharmacy — Interactive Viewer")

#############################################
# Load Model
#############################################
st.sidebar.header("🔍 Model Loader")

uploaded_model = st.sidebar.file_uploader("Upload GPIPD Model (.tar)", type=["tar"])

if uploaded_model:
    # Save temp
    model_path = "uploaded_model.tar"
    with open(model_path, "wb") as f:
        f.write(uploaded_model.read())

    st.sidebar.success("Model uploaded!")

#############################################
# Initialize environment
#############################################
env = PolypharmacyEnv()

#############################################
# Load Agent (only if model available)
#############################################
if uploaded_model:
    st.sidebar.write("Initializing Agent...")
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
    agent.load(model_path)
    st.sidebar.success("Model loaded!")

#############################################
# Helper: Predict Using RL Policy
#############################################
def agent_predict(obs, w):
    return agent.eval(obs, w)

#############################################
# Section 1 — Weight Control Panel
#############################################
st.header("⚖️ Weight Selection for Multi-Objective Tradeoffs")

w1 = st.slider("Efficacy weight", 0.0, 1.0, 0.33)
w2 = st.slider("DDI weight", 0.0, 1.0, 0.33)
w3 = st.slider("Tolerability weight", 0.0, 1.0, 0.34)

weight_vector = np.array([w1, w2, w3])
weight_vector /= weight_vector.sum()

st.write(f"Active weight vector: **{weight_vector}**")

#############################################
# Section 2 — Simulate Episode
#############################################
st.header("🎬 Simulate Prescription Strategy")

if uploaded_model:
    if st.button("Run 1 Episode"):
        obs, _ = env.reset()
        total_reward = np.zeros(3)

        steps = []
        done = False
        while not done:
            action = agent_predict(obs, weight_vector)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            steps.append({
                "action": int(action),
                "reward_efficacy": reward[0],
                "reward_ddi": reward[1],
                "reward_tol": reward[2],
                "hadm_id": info["hadm_id"]
            })

        st.subheader("📊 Episode Summary")
        st.json({
            "total_reward": total_reward.tolist(),
            "steps": steps
        })

#############################################
# Section 3 — Visualizations
#############################################
st.header("📈 Visualizations")

uploaded_json = st.file_uploader("Upload Phase 8 JSON (optional)", type=["json"])

if uploaded_json:
    data = json.load(uploaded_json)

    st.subheader("Pareto Frontier (Mean)")
    if "pareto_plot" in data:
        st.image(data["pareto_plot"])

    st.subheader("Action Frequency Heatmap")
    if "heatmap" in data:
        st.image(data["heatmap"])

#############################################
st.markdown("---")
st.write("🔥 **Powered by MORL-Baselines + Streamlit**")
