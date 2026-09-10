import os
import pickle
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from models.lstm_world_model import LSTMWorldModel
from features.network_state import STATE_FEATURES, create_network_states
from forecasting.k_step_forecast import forecast_k_steps
from forecasting.risk_trajectory import calculate_risk_trajectory
from explainability.mitre_mapping import evaluate_mitre_context
from explainability.feature_attribution import calculate_feature_attribution

class InferenceService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
    def load_artifacts(self):
        if self.is_loaded:
            return
            
        model_path = "models_weights/lstm_world_model.pth"
        scaler_path = "models_weights/network_state_scaler.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler weights not found. Cannot start inference service.")
            
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        self.model = LSTMWorldModel(input_size=len(STATE_FEATURES), hidden_size=64, num_layers=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        self.is_loaded = True
        
    def process_dataframe(self, df: pd.DataFrame, k_steps: int = 5) -> Dict[str, Any]:
        """
        Processes a raw DataFrame, extracts network states, runs inference and forecasting.
        """
        if not self.is_loaded:
            self.load_artifacts()
            
        # Clean columns
        df.columns = df.columns.str.strip()
        
        # Check required columns
        missing_cols = [c for c in STATE_FEATURES if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required state features: {missing_cols}")
            
        state_df = df[STATE_FEATURES].copy()
        for col in STATE_FEATURES:
            state_df[col] = pd.to_numeric(state_df[col], errors="coerce")
        state_df = state_df.replace([np.inf, -np.inf], np.nan)
        
        valid_state_mask = state_df.notna().all(axis=1) & (state_df >= 0).all(axis=1)
        state_df = state_df.loc[valid_state_mask].copy()
        
        network_states = create_network_states(state_df)
        
        sequence_length = 10
        if len(network_states) < sequence_length:
            raise ValueError(f"Not enough valid rows to form a {sequence_length}-step sequence. (Found {len(network_states)})")
            
        # Take the most recent sequence
        initial_raw_sequence = network_states[-sequence_length:]
        initial_scaled_sequence = self.scaler.transform(initial_raw_sequence).astype(np.float32)
        
        # 1. K-Step Forecast
        forecast_result = forecast_k_steps(
            model=self.model,
            initial_sequence=initial_scaled_sequence,
            scaler=self.scaler,
            k_steps=k_steps,
            device=self.device
        )
        
        # 2. Current state probability (for risk trajectory)
        seq_tensor = torch.tensor(initial_scaled_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, current_attack_logits = self.model(seq_tensor)
            current_probability = torch.sigmoid(current_attack_logits).item()
            
        # 3. Risk Trajectory
        risk_result = calculate_risk_trajectory(
            current_probability=current_probability,
            future_probabilities=forecast_result["attack_probabilities"]
        )
        
        # 4. MITRE Context (using the latest raw state)
        latest_raw_state = initial_raw_sequence[-1]
        features_dict = {feat: float(val) for feat, val in zip(STATE_FEATURES, latest_raw_state)}
        mitre_context = evaluate_mitre_context(features_dict)
        
        # 5. Explainability (Feature Attribution)
        explainability = calculate_feature_attribution(self.model, seq_tensor, self.device)
        
        return {
            "current_probability": current_probability,
            "current_risk_level": risk_result["current_risk"]["risk_level"],
            "forecast_windows": risk_result["forecast_windows"],
            "summary": risk_result["summary"],
            "mitre_context": mitre_context,
            "explainability": explainability,
            "raw_features": features_dict,
            "future_states": [
                {feat: float(val) for feat, val in zip(STATE_FEATURES, state)} 
                for state in forecast_result["future_states"]
            ]
        }

# Global singleton
inference_service = InferenceService()
