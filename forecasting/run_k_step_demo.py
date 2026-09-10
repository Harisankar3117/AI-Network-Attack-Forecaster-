import pickle
import numpy as np
import pandas as pd
import torch

from features.network_state import STATE_FEATURES, create_network_states
from models.lstm_world_model import LSTMWorldModel
from forecasting.k_step_forecast import forecast_k_steps

def run_demo():
    print("=" * 50)
    print("K-STEP WORLD MODEL FORECAST")
    print("=" * 50)
    
    # Load components silently
    with open("models_weights/network_state_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    model = LSTMWorldModel(input_size=18, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load("models_weights/lstm_world_model.pth", map_location="cpu", weights_only=True))
    model.eval()
    
    df = pd.read_csv("data/processed/CIC-IDS2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter_cleaned.csv", low_memory=False, nrows=500)
    df.columns = df.columns.str.strip()
    
    state_df = df[STATE_FEATURES].copy()
    for col in STATE_FEATURES:
        state_df[col] = pd.to_numeric(state_df[col], errors="coerce")
    state_df = state_df.replace([np.inf, -np.inf], np.nan)
    valid_state_mask = state_df.notna().all(axis=1) & (state_df >= 0).all(axis=1)
    state_df = state_df.loc[valid_state_mask].copy()
    
    network_states = create_network_states(state_df)
    initial_raw_sequence = network_states[:10]
    initial_scaled_sequence = scaler.transform(initial_raw_sequence).astype(np.float32)
    
    print("\nRunning forecast for K=5 steps into the future...\n")
    
    result = forecast_k_steps(
        model=model,
        initial_sequence=initial_scaled_sequence,
        scaler=scaler,
        k_steps=5,
        device="cpu"
    )
    
    for i in range(5):
        print(f"Step {result['steps'][i]}:")
        print(f"Attack Probability: {result['attack_probabilities'][i] * 100:.2f}%")
        print(f"Risk: {result['risk_levels'][i].upper()}\n")
        
    print("Forecast complete.")

if __name__ == "__main__":
    run_demo()
