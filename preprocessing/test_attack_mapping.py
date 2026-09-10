import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from forecasting.attack_mapping import (
    generate_attack_stage_forecast
)


attack_probabilities = [
    0.21,
    0.34,
    0.52,
    0.71,
    0.86
]


forecast = generate_attack_stage_forecast(
    attack_probabilities
)


print("\n🎯 MITRE ATT&CK Stage Forecast")
print("=" * 55)

for item in forecast:

    print(
        f"Future Step {item['future_step']} | "
        f"Attack Probability: "
        f"{item['attack_probability']:.2%} | "
        f"Predicted Stage: "
        f"{item['predicted_stage']}"
    )

print(
    "\n✅ Attack stage forecasting completed successfully!"
)