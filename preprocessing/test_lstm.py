import torch

from models.lstm_world_model import LSTMWorldModel


# ============================================================
# CREATE LSTM WORLD MODEL
# ============================================================

model = LSTMWorldModel(
    input_size=18,
    hidden_size=64,
    num_layers=2
)

print("LSTM World Model created successfully!")


# ============================================================
# PRINT MODEL
# ============================================================

print("\nModel:")
print(model)


# ============================================================
# CREATE DUMMY TEMPORAL INPUT
# ============================================================

# Batch size       = 2
# Sequence length  = 10
# Network features = 18

x = torch.randn(
    2,
    10,
    18
)

print("\nInput Shape:")
print(x.shape)


# ============================================================
# RUN MODEL
# ============================================================

next_state, _ = model(x)
attack_probability = model.predict_attack_probability(x)


# ============================================================
# OUTPUT SHAPES
# ============================================================

print("\nNext State Shape:")
print(next_state.shape)

print("\nAttack Probability Shape:")
print(attack_probability.shape)


# ============================================================
# ATTACK PROBABILITY
# ============================================================

print("\nAttack Probability:")
print(attack_probability)


# ============================================================
# VALIDATION
# ============================================================

assert x.shape == (2, 10, 18)
assert next_state.shape == (2, 18)
assert attack_probability.shape == (2, 1)

assert torch.all(
    attack_probability >= 0
)

assert torch.all(
    attack_probability <= 1
)


print("\n" + "=" * 70)
print("LSTM MODEL TEST PASSED SUCCESSFULLY!")
print("=" * 70)