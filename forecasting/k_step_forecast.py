import torch
import numpy as np

def forecast_k_steps(model, initial_sequence, scaler=None, k_steps=5, device="cpu"):
    """
    Recursively predicts the next K states and attack probabilities.

    Args:
        model (LSTMWorldModel): Trained PyTorch world model.
        initial_sequence (np.ndarray or torch.Tensor): The scaled initial sequence.
            Shape must be [sequence_length, input_features] or [1, sequence_length, input_features].
        scaler (sklearn.preprocessing.StandardScaler, optional): Fitted scaler to inverse transform predictions.
        k_steps (int, optional): Number of steps to forecast into the future. Defaults to 5.
        device (str, optional): Device to run inference on. Defaults to "cpu".

    Returns:
        dict: A dictionary containing:
            - "future_states": numpy array of raw unscaled predicted states of shape (k_steps, input_features)
            - "attack_probabilities": list of attack probabilities for each step
            - "risk_levels": list of risk level strings for each step
            - "steps": list of step integers [1, 2, ..., k_steps]
    """
    
    # 1. Input Validation and Preparation
    model.eval()
    
    if isinstance(initial_sequence, np.ndarray):
        seq = torch.tensor(initial_sequence, dtype=torch.float32)
    else:
        seq = initial_sequence.clone().detach().float()
        
    if seq.dim() == 2:
        seq = seq.unsqueeze(0)  # Add batch dimension -> [1, seq_len, features]
    elif seq.dim() != 3 or seq.shape[0] != 1:
        raise ValueError(f"Expected sequence of shape [seq_len, features] or [1, seq_len, features], got {seq.shape}")
        
    seq = seq.to(device)
    model = model.to(device)
    
    # Storage for predictions
    predicted_scaled_states = []
    attack_probabilities = []
    risk_levels = []
    steps = []
    
    current_seq = seq
    
    # 2. Recursive K-Step Forecast Loop
    with torch.no_grad():
        for step in range(1, k_steps + 1):
            
            # Forward pass: model expects [batch, seq_len, features]
            next_state_scaled, attack_logits = model(current_seq)
            
            # next_state_scaled: [1, features]
            # attack_logits: [1, 1]
            
            # Calculate probability using sigmoid
            prob = torch.sigmoid(attack_logits).item()
            
            # Determine Risk Level
            if prob < 0.30:
                risk = "Low"
            elif prob < 0.70:
                risk = "Medium"
            else:
                risk = "High"
                
            # Store results
            predicted_scaled_states.append(next_state_scaled.squeeze(0).cpu().numpy())
            attack_probabilities.append(prob)
            risk_levels.append(risk)
            steps.append(step)
            
            # Recursive step: Update the sliding window
            # Remove the oldest state (index 0) and append the newly predicted state
            next_state_expanded = next_state_scaled.unsqueeze(1)  # [1, 1, features]
            current_seq = torch.cat([current_seq[:, 1:, :], next_state_expanded], dim=1)
            
    # 3. Output Processing
    predicted_scaled_states = np.array(predicted_scaled_states) # Shape: (k_steps, features)
    
    if scaler is not None:
        # Inverse transform the scaled predictions to get raw interpretable state values
        future_states = scaler.inverse_transform(predicted_scaled_states)
    else:
        future_states = predicted_scaled_states

    return {
        "future_states": future_states,
        "attack_probabilities": attack_probabilities,
        "risk_levels": risk_levels,
        "steps": steps
    }
