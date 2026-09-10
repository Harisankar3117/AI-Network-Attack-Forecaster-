def get_risk_level(probability):

    if probability < 0.30:
        return "LOW"

    elif probability < 0.60:
        return "MEDIUM"

    elif probability < 0.80:
        return "HIGH"

    else:
        return "CRITICAL"


def generate_risk_forecast(attack_probabilities):

    forecast = []

    for step, probability in enumerate(attack_probabilities, start=1):

        risk_level = get_risk_level(probability)

        forecast.append({
            "future_step": step,
            "attack_probability": probability,
            "risk_level": risk_level
        })

    return forecast