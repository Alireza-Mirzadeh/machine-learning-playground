# Implement Positional Encoding for Transformers (sin/cos)

import numpy as np


def postional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Generate positional encodings for a sequence of length `seq_len` and model dimension `d_model`.

    Args:
        seq_len (int): Length of the sequence.
        d_model (int): Dimension of the model.
        base (float): Base for the exponential function (default is 10000.0).

    Returns:
        np.ndarray: A matrix of shape (seq_len, d_model) containing the positional encodings.

    """

    # Create a matrix of shape (seq_len, d_model) to hold the positional encodings
    pos_enc = np.zeros((seq_len, d_model))

    # Calculate the positional encodings
    for pos in range(seq_len):
        for i in range(d_model):
            angle = pos / np.power(base, (2 * (i // 2)) / d_model)

            if i % 2 == 0:
                pos_enc[pos, i] = np.sin(angle)
            else:
                pos_enc[pos, i] = np.cos(angle)

    return pos_enc


# Example usage
if __name__ == "__main__":
    seq_len = 5  # Length of the sequence
    d_model = 7  # Dimension of the model
    pos_encodings = postional_encoding(seq_len, d_model)
    print(pos_encodings)
    print("Shape of positional encodings:", pos_encodings.shape)
