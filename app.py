import torch
import streamlit as st
import numpy as np
from polypharmacy_env import PolypharmacyEnv

# Load model file
MODEL_PATH = "gpipd_model_latest.tar"

st.title("Polypharmacy MORL Policy Viewer")

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    q_nets = []

    # Load saved Q-networks
    for i in range(2):  # GPIPD has 2 psi nets
        key = f"psi_net_{i}_state_dict"
        if key in checkpoint:
            q_net = build_q_net()               # We'll define this next
            q_net.load_state_dict(checkpoint[key])
            q_nets.append(q_net)

    weight_support = checkpoint["M"]

    return q_nets, weight_support


def build_q_net():
    # Minimal network architecture identical to training
    return torch.nn.Sequential(
        torch.nn.Linear(4 + 3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2 * 3)  # action_dim * reward_dim
    )


def eval_action(q_nets, obs, w):
    obs = torch.tensor(obs, dtype=torch.float32)
    w = torch.tensor(w, dtype=torch.float32)

    psi_values = q_nets[0](torch.cat([obs, w]))
    psi_values = psi_values.view(2, 3)       # reshape to (num_actions, reward_dim)

    # scalarize
    scalar_q = torch.matmul(psi_values, w)
    action = torch.argmax(scalar_q).item()
    return action


# Load model
q_nets, weight_support = load_model()
st.success("Model loaded successfully!")

env = PolypharmacyEnv()

# UI
weight = st.slider("Tradeoff (w1 = efficacy, w2 = DDI, w3 = tolerability)", 0.0, 1.0, 0.33)
w = np.array([weight, 1 - weight, 0.1])

if st.button("Predict Action"):
    obs, _ = env.reset()
    action = eval_action(q_nets, obs, w)
    st.write("Recommended Action:", action)
