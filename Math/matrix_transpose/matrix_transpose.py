# Function to transpose a matrix represented as a list of lists and return it as a NumPy array.

import numpy as np


def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.

    Args:
        A (list): A list of lists representing the input matrix.

    Returns:
        np.ndarray: A NumPy array representing the transposed matrix.
    """

    # Check if the input matrix is empty
    if not A:
        print("Input matrix is empty. Returning an empty array.")
        return np.array([])

    rows = len(A)
    cols = len(A[0])

    # Create a new matrix with dimensions cols x rows for the transposed matrix
    T = np.empty((cols, rows), dtype=np.array(A).dtype)

    # Loop through the original matrix and fill in the transposed matrix
    for i in range(rows):
        for j in range(cols):
            T[j, i] = A[i][j]

    return T


# Test the transpose function with a sample matrix
if __name__ == "__main__":
    A = [[1, 2, 3], [4, 5, 6]]
    print(f"Original matrix: \n {np.array(A)}")

    T_A = matrix_transpose(A)
    print(f"Transposed matrix: \n {T_A}")
