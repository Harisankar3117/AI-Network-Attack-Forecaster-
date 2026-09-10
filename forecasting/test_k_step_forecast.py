import os
import pickle
import numpy as np
import pandas as pd
import torch

from features.network_state import STATE_FEATURES, create_network_states
from models.lstm_world_model import LSTMWorldModel
from forecasting.k_step_forecast import forecast_k_steps

def main():
    print("=" * 50)
    print("TESTING K-STEP FUTURE WORLD-MODEL ROLLOUT")
    print("=" * 50)
    
    # Paths
    MODEL_PATH = "models_weights/lstm_world_model.pth"
    SCALER_PATH = "models_weights/network_state_scaler.pkl"
    DATA_PATH = "data/processed/CIC-IDS2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter_cleaned.csv"
    
    # 1 & 2. Load Model and Scaler
    print(f"Loading Scaler from {SCALER_PATH}...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
        
    print(f"Loading LSTM World Model from {MODEL_PATH}...")
    model = LSTMWorldModel(input_size=len(STATE_FEATURES), hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    
    # 3. Load Dataset
    print(f"Loading sample data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False, nrows=500) # Only need a small valid sequence
    df.columns = df.columns.str.strip()
    
    # 4. Build 18-feature network state
    state_df = df[STATE_FEATURES].copy()
    for col in STATE_FEATURES:
        state_df[col] = pd.to_numeric(state_df[col], errors="coerce")
    state_df = state_df.replace([np.inf, -np.inf], np.nan)
    
    valid_state_mask = state_df.notna().all(axis=1) & (state_df >= 0).all(axis=1)
    state_df = state_df.loc[valid_state_mask].copy()
    
    network_states = create_network_states(state_df)
    
    # 5. Select a valid 10-step sequence
    SEQUENCE_LENGTH = 10
    if len(network_states) < SEQUENCE_LENGTH:
        raise ValueError("Not enough valid states to form a sequence.")
        
    initial_raw_sequence = network_states[:SEQUENCE_LENGTH]
    
    # 6. Scale it exactly as the trained model expects
    initial_scaled_sequence = scaler.transform(initial_raw_sequence).astype(np.float32)
    
    # 7. Run K=5
    K = 5
    print(f"\nRunning K-Step Forecast (K={K})...")
    result = forecast_k_steps(
        model=model,
        initial_sequence=initial_scaled_sequence,
        scaler=scaler,
        k_steps=K,
        device="cpu"
    )
    
    future_states = result["future_states"]
    probs = result["attack_probabilities"]
    risk_levels = result["risk_levels"]
    
    # 8. Print Output
    print("\n--- RESULTS ---")
    print(f"Initial sequence shape: {initial_scaled_sequence.shape}")
    print(f"Future state shape: {future_states.shape}")
    print(f"Future probabilities: {[round(p, 4) for p in probs]}")
    print(f"Risk levels: {risk_levels}")
    print(f"Minimum probability: {min(probs):.4f}")
    print(f"Maximum probability: {max(probs):.4f}")
    print(f"Average probability: {np.mean(probs):.4f}")
    
    # 9. Verify Assertions
    print("\n--- VERIFICATION ---")
    
    assert future_states.shape == (K, 18), f"Expected shape (5, 18), got {future_states.shape}"
    print("[OK] future_states.shape == (5, 18)")
    
    assert len(probs) == K, f"Expected length {K}, got {len(probs)}"
    print("[OK] attack_probabilities length == 5")
    
    assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of bounds [0, 1]"
    print("[OK] all probabilities are between 0 and 1")
    
    assert not np.isnan(future_states).any(), "NaN found in future states"
    assert not np.isnan(probs).any(), "NaN found in probabilities"
    print("[OK] no NaN values")
    
    assert not np.isinf(future_states).any(), "Infinity found in future states"
    assert not np.isinf(probs).any(), "Infinity found in probabilities"
    print("[OK] no infinity values")
    
    print("\nSUCCESS: All K-Step World Model rollout tests passed perfectly.")

if __name__ == "__main__":
    main()
