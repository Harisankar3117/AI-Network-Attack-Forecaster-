import torch
import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.lstm_world_model import LSTMWorldModel
from forecasting.rollout import k_step_forecast
from forecasting.risk_forecast import generate_risk_forecast
from forecasting.attack_mapping import generate_attack_stage_forecast


# ============================================================
# 1. Load trained LSTM model
# ============================================================

model = LSTMWorldModel(
    input_size=18,
    hidden_size=64,
    num_layers=2
)

model.load_state_dict(
    torch.load(
        "models_weights/lstm_world_model.pth",
        weights_only=True
    )
)

model.eval()

print("✅ Trained LSTM model loaded successfully!")


# ============================================================
# 2. Current network state sequence
# ============================================================

initial_sequence = torch.randn(1, 3, 18, dtype=torch.float32)


# ============================================================
# 3. K-Step Forecast
# ============================================================

predicted_states, attack_probabilities = k_step_forecast(
    model,
    initial_sequence,
    k=5
)


# ============================================================
# 4. Generate Risk Forecast
# ============================================================

risk_forecast = generate_risk_forecast(
    attack_probabilities
)


# ============================================================
# 5. Generate Attack Stage Forecast
# ============================================================

stage_forecast = generate_attack_stage_forecast(
    attack_probabilities
)


# ============================================================
# 6. Display Final Forecast
# ============================================================

print("\n")
print("=" * 75)
print("🔮 AI NETWORK ATTACK FORECAST")
print("=" * 75)

for i in range(5):

    probability = attack_probabilities[i]

    risk = risk_forecast[i]["risk_level"]

    stage = stage_forecast[i]["predicted_stage"]

    print(f"\nFuture Step {i + 1}")
    print("-" * 75)

    print(
        f"Attack Probability : {probability:.2%}"
    )

    print(
        f"Risk Level         : {risk}"
    )

    print(
        f"Predicted Stage    : {stage}"
    )

    print(
        f"Predicted State    : {predicted_states[i].numpy()}"
    )


print("\n")
print("=" * 75)
print("✅ INTEGRATED FORECASTING PIPELINE COMPLETED!")
print("=" * 75)