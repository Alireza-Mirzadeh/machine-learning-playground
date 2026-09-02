## Matrix Transpose

The **transpose** of a matrix changes its **rows into columns** and its **columns into rows**.

For example:

```text
Original (2 × 3)       Transpose (3 × 2)

1  2  3                1  4
4  5  6       →        2  5
                       3  6
```

In Python:

```python
rows = len(A)       # number of rows
cols = len(A[0])    # number of columns
```

To transpose, each element at position `(i, j)` moves to `(j, i)`:

```python
T[j, i] = A[i][j]
```

**Key idea:** `(row, column) → (column, row)`.
