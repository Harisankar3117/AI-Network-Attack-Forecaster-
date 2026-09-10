import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from forecasting.risk_forecast import generate_risk_forecast


# Example probabilities from the LSTM
attack_probabilities = [
    0.21,
    0.34,
    0.52,
    0.71,
    0.86
]


forecast = generate_risk_forecast(
    attack_probabilities
)


print("\n🔮 Future Risk Forecast")
print("=" * 45)

for item in forecast:

    print(
        f"Future Step {item['future_step']} | "
        f"Attack Probability: "
        f"{item['attack_probability']:.2%} | "
        f"Risk: {item['risk_level']}"
    )

print("\n✅ Risk forecasting completed successfully!")