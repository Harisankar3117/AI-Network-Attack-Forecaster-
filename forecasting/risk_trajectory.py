from typing import List, Dict, Any, Tuple

# Risk bands based on project documentation
RISK_BANDS = {
    "Low": (0.00, 0.2499),
    "Medium": (0.25, 0.4999),
    "High": (0.50, 0.7499),
    "Critical": (0.75, 1.00)
}

def get_risk_level(probability: float) -> str:
    """Returns the risk level based on the given probability."""
    if probability < 0.25:
        return "Low"
    elif probability < 0.50:
        return "Medium"
    elif probability < 0.75:
        return "High"
    else:
        return "Critical"

def calculate_risk_trajectory(current_probability: float, future_probabilities: List[float]) -> Dict[str, Any]:
    """
    Calculates the risk trajectory, peak risk, average risk, trends, and early warnings
    based on the current probability and future forecasted K-step probabilities.
    
    Args:
        current_probability (float): The attack probability of the current network state S(t).
        future_probabilities (List[float]): The attack probabilities of future states S(t+1)...S(t+K).
        
    Returns:
        Dict[str, Any]: A detailed risk trajectory and summary.
    """
    windows = []
    
    peak_risk_prob = current_probability
    sum_risk_prob = current_probability
    
    earliest_high_risk_window = None
    
    prev_prob = current_probability
    
    for i, prob in enumerate(future_probabilities):
        step = i + 1
        
        # Risk level
        level = get_risk_level(prob)
        
        # Risk change
        change = prob - prev_prob
        
        # Early warning status
        is_early_warning = level in ["High", "Critical"]
        
        # Check if it's the first high/critical risk
        if is_early_warning and earliest_high_risk_window is None:
            earliest_high_risk_window = step
            
        windows.append({
            "window_index": step,
            "probability": prob,
            "risk_level": level,
            "risk_change": change,
            "early_warning": is_early_warning
        })
        
        if prob > peak_risk_prob:
            peak_risk_prob = prob
            
        sum_risk_prob += prob
        prev_prob = prob
        
    # Summaries
    avg_risk = sum_risk_prob / (len(future_probabilities) + 1) if future_probabilities else current_probability
    
    # Risk trend
    if not future_probabilities:
        trend = "Stable"
    else:
        final_prob = future_probabilities[-1]
        if final_prob - current_probability >= 0.10:
            trend = "Increasing"
        elif current_probability - final_prob >= 0.10:
            trend = "Decreasing"
        else:
            trend = "Stable"
            
    # Warning message
    if earliest_high_risk_window is not None:
        warning_message = f"High risk forecasted in {earliest_high_risk_window} future forecast windows."
    elif trend == "Increasing" and peak_risk_prob >= 0.25:
        warning_message = "Risk is increasing towards Medium."
    else:
        warning_message = "Network conditions are currently stable."

    return {
        "current_risk": {
            "probability": current_probability,
            "risk_level": get_risk_level(current_probability)
        },
        "forecast_windows": windows,
        "summary": {
            "peak_forecast_risk_probability": peak_risk_prob,
            "peak_forecast_risk_level": get_risk_level(peak_risk_prob),
            "average_forecast_risk_probability": avg_risk,
            "average_forecast_risk_level": get_risk_level(avg_risk),
            "risk_trend": trend,
            "earliest_high_risk_window": earliest_high_risk_window,
            "predicted_attack_count": sum(1 for w in windows if w["early_warning"]),
            "warning_message": warning_message
        }
    }
