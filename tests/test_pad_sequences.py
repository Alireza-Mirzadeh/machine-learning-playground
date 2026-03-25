# Test cases for the pad_sequences function in the NLP module.

from NLP import pad_sequences
import numpy as np

def test_pad_sequences():
    """
    Test the pad_sequences function with various inputs.
    """
    # Test case 1: Basic padding
    seqs1 = [[1, 2, 2], [3, 4], [5]]
    
    expected_output1 = [[1, 2, 2], [3, 4, 0], [5, 0, 0]]
    
    output1 = pad_sequences(seqs1, pad_value=0)
    
    assert np.array_equal(output1, expected_output1), f"Expected {expected_output1}, but got {output1}"
    
    # Test case 2: Different padding value
    seqs2 = [[1, 2, 2], [3, 4], [5]]
    
    expected_output2 = [[1, 2, 2], [3, 4, -1], [5, -1, -1]]
    
    output2 = pad_sequences(seqs2, pad_value=-1)
    
    assert np.array_equal(output2, expected_output2), f"Expected {expected_output2}, but got {output2}"
    
    # Test case 3: Empty sequences
    seqs3 = [[], [1, 2], [3]]

    expected_output3 = [[0, 0], [1, 2], [3, 0]]
    
    output3 = pad_sequences(seqs3, pad_value=0)
    
    assert np.array_equal(output3, expected_output3), f"Expected {expected_output3} but got {output3}"
    
    # Test case 4: Max length specified
    seqs4 = [[1, 2, 3, 4, 5], [6, 7, 9], [10]]
    
    expected_output4 = [[1, 2, 3], [6, 7, 9], [10, 0, 0]]
    
    output4 = pad_sequences(seqs4, max_len=3, pad_value=0)
    
    assert np.array_equal(output4, expected_output4), f"Expected {expected_output4} but got {output4}"

    