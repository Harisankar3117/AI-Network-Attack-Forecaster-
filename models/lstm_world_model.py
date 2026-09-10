import torch
import torch.nn as nn


class LSTMWorldModel(nn.Module):
    """
    LSTM-based Network World Model.

    Input:
        [batch_size, sequence_length, input_size]

    Output:
        next_state:
            [batch_size, input_size]

        attack_logits:
            [batch_size, 1]
    """

    def __init__(
        self,
        input_size=18,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ----------------------------------------------------
        # Temporal Encoder
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ----------------------------------------------------
        # State Transition Head
        # ----------------------------------------------------

        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

        # ----------------------------------------------------
        # Attack Risk Head
        # ----------------------------------------------------

        self.attack_predictor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        # x:
        # [batch, sequence_length, features]

        output, _ = self.lstm(x)

        # Last temporal representation
        last_output = output[:, -1, :]

        # Predict next network state
        next_state = self.state_predictor(
            last_output
        )

        # Predict attack logit
        # NOTE:
        # No sigmoid here.
        # BCEWithLogitsLoss will handle it.
        attack_logits = self.attack_predictor(
            last_output
        )

        return next_state, attack_logits

    def predict_attack_probability(self, x):

        _, attack_logits = self.forward(x)

        return torch.sigmoid(
            attack_logits
        )