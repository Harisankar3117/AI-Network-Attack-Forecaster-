import torch


def k_step_forecast(model, initial_sequence, k=5):
    """
    Forecast future network states for K steps.

    Args:
        model: Trained LSTM World Model
        initial_sequence: Tensor of shape [1, sequence_length, features]
        k: Number of future steps

    Returns:
        predicted_states: List of predicted future states
        attack_probabilities: List of predicted attack probabilities
    """

    model.eval()

    current_sequence = initial_sequence.clone()

    predicted_states = []
    attack_probabilities = []

    with torch.no_grad():

        for step in range(k):

            # Predict next state and attack probability
            next_state, attack_probability = model(current_sequence)

            # Store prediction
            predicted_states.append(next_state.squeeze(0))
            attack_probabilities.append(attack_probability.item())

            # Add predicted state to sequence
            next_state = next_state.unsqueeze(1)

            # Remove oldest state and append new predicted state
            current_sequence = torch.cat(
                [current_sequence[:, 1:, :], next_state],
                dim=1
            )

    return predicted_states, attack_probabilities