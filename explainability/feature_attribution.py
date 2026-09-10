import torch
import numpy as np
from typing import List, Dict, Any

from features.network_state import STATE_FEATURES

def calculate_feature_attribution(model: torch.nn.Module, sequence: torch.Tensor, device: str = "cpu") -> List[Dict[str, Any]]:
    """
    Calculates feature importance using Input x Gradient attribution.
    This provides an honest, mathematically grounded explanation for the 
    LSTM's attack logit prediction based on the input sequence.

    Args:
        model: The trained PyTorch LSTMWorldModel.
        sequence: The scaled input sequence tensor of shape [1, seq_len, num_features].
        device: The device on which the model and tensor reside.

    Returns:
        List[Dict[str, Any]]: Top contributing features sorted by absolute importance.
    """
    model.eval()
    
    # Ensure sequence requires gradients
    input_tensor = sequence.clone().detach().to(device)
    input_tensor.requires_grad_(True)
    
    # Forward pass
    _, attack_logits = model(input_tensor)
    
    # Clear previous gradients
    model.zero_grad()
    
    # Backward pass to compute gradients with respect to input
    attack_logits.backward(torch.ones_like(attack_logits))
    
    # Extract gradients (shape: [1, seq_len, num_features])
    gradients = input_tensor.grad
    
    if gradients is None:
        raise ValueError("Gradients could not be computed. Ensure the model supports autograd on input.")
        
    # Calculate Input x Gradient
    # attribution = input * gradient
    attribution = input_tensor * gradients
    
    # Aggregate attribution across the temporal sequence (sum over sequence length)
    # Shape becomes [1, num_features] -> [num_features]
    temporal_attribution = attribution.sum(dim=1).squeeze(0).cpu().detach().numpy()
    
    # Extract raw input sum across time for context
    temporal_input = input_tensor.sum(dim=1).squeeze(0).detach().cpu().numpy()
    
    results = []
    for i, feature_name in enumerate(STATE_FEATURES):
        attr_val = float(temporal_attribution[i])
        
        # Determine direction: 
        # Positive attribution means this feature increased the attack probability.
        # Negative means it decreased it.
        direction = "Increases Risk" if attr_val > 0 else "Decreases Risk"
        if abs(attr_val) < 1e-5:
            direction = "Neutral"
            
        results.append({
            "feature": feature_name,
            "importance": abs(attr_val),
            "raw_attribution": attr_val,
            "direction": direction,
            "explanation": f"Input x Gradient attribution over {sequence.shape[1]} timesteps."
        })
        
    # Sort by absolute importance descending
    results.sort(key=lambda x: x["importance"], reverse=True)
    
    return results
