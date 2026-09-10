import torch
import sys
import os

# Allow importing the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_world_model import LSTMWorldModel
from forecasting.rollout import k_step_forecast


# Load trained model
model = LSTMWorldModel(
    input_size=18,
    hidden_size=64,
    num_layers=2
)

model.load_state_dict(
    torch.load("models_weights/lstm_world_model.pth")
)

model.eval()

print("✅ Trained LSTM model loaded successfully!")


# Create a sample initial sequence
# Shape: [batch_size, sequence_length, features]
initial_sequence = torch.randn(1, 3, 18, dtype=torch.float32)


# Perform 5-step forecasting
predicted_states, attack_probabilities = k_step_forecast(
    model,
    initial_sequence,
    k=5
)


print("\n🔮 K-Step Future Forecast")
print("=" * 40)

for i in range(5):

    print(f"\nFuture Step {i + 1}")

    print("Predicted Network State:")
    print(predicted_states[i])

    print(
        f"Attack Probability: "
        f"{attack_probabilities[i]:.4f}"
    )

print("\n✅ K-step forecasting completed successfully!")