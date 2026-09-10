import numpy as np


def create_sequences(data, sequence_length=10):
    """
    Create temporal input sequences and next-state targets.

    Example:
        States 1-10  -> Target State 11
        States 2-11  -> Target State 12
        States 3-12  -> Target State 13

    Args:
        data: NumPy array of network states
        sequence_length: Number of past states used as input

    Returns:
        X: Input sequences
        y: Next-state targets
    """

    print("\nCreating temporal sequences...")

    data = np.asarray(data, dtype=np.float32)

    if len(data) <= sequence_length:
        raise ValueError(
            "Not enough data to create sequences."
        )

    X = []
    y = []

    for i in range(
        len(data) - sequence_length
    ):
        sequence = data[
            i:i + sequence_length
        ]

        next_state = data[
            i + sequence_length
        ]

        X.append(sequence)
        y.append(next_state)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    print("Temporal sequences created!")
    print("--------------------------------")
    print("Sequence length :", sequence_length)
    print("Input shape     :", X.shape)
    print("Target shape    :", y.shape)
    print("--------------------------------")

    return X, y