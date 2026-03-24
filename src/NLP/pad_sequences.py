import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Pad sequences to the same length.
    
    Args:
        seqs: List of sequences (lists of integers).
        pad_value: value to use for padding (defult: 0).
        max_len: maximum length to pad to (default: None, which means pad to the length of the longest sequence).
    
    Returns:
        np.ndarray of shape (N, L) where:
        N = len(seqs)
        L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Handle the case where seqs is empty
    if not seqs:
        return np.empty((0,0))
    
    # Determine the length to pad to based on max_len or the longest sequence
    padding_length = (max_len if max_len is not None
                    else max(len(seq) for seq in seqs))
    
    # Empty list to hold the padded sequences
    result = []

    # Pad each sequence in seqs to the determined padding_length
    for seq in seqs:
        
        # If the sequence is shorter than the padding_length, pad it with pad_value until it reaches padding_length   
        if len(seq) < padding_length:
            n_to_add = padding_length - len(seq)
            new_seq = seq +[pad_value] * n_to_add
        
        # If the sequence is longer than the padding_length, truncate it to fit the padding_length
        else:
            new_seq = seq[:padding_length]

        # Append the padded (or truncated) sequence to the result list
        result.append(new_seq)

    # Convert the list of padded sequences to a numpy array and return it
    return np.array(result)

# Demo
if __name__ == "__main__":
    seqs = [[1, 2, 3], [4, 5], [6]]
    padded = pad_sequences(seqs, pad_value=0, max_len=None)
    print(padded)
