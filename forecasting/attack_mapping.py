def predict_attack_stage(probability):
    """
    Map predicted attack probability to a high-level
    MITRE ATT&CK-related attack stage.
    """

    if probability < 0.30:
        return "Normal / No Threat"

    elif probability < 0.60:
        return "Reconnaissance"

    elif probability < 0.80:
        return "Initial Access"

    else:
        return "Execution / Lateral Movement"


def generate_attack_stage_forecast(attack_probabilities):

    forecast = []

    for step, probability in enumerate(
        attack_probabilities,
        start=1
    ):

        stage = predict_attack_stage(probability)

        forecast.append({
            "future_step": step,
            "attack_probability": probability,
            "predicted_stage": stage
        })

    return forecast