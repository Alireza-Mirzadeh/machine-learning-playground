## Positional Encoding

### What is it?

A Transformer processes all tokens at the same time, so it does not naturally know the **order of the tokens**.

Positional encoding adds information about **where each token is located in the sequence**.

For example:

```text
"I"    → position 0
"love" → position 1
"AI"   → position 2
```

Each position receives a unique **pattern of numbers**.

---

### How does it work?

We create a matrix:

```python
pos_enc = np.zeros((seq_len, d_model))
```

* `seq_len` → number of tokens/positions
* `d_model` → number of values used to represent each position

For example:

```text
seq_len = 3
d_model = 4

Position     Positional encoding
   0         [ ... ... ... ... ]
   1         [ ... ... ... ... ]
   2         [ ... ... ... ... ]
```

Each position gets a different pattern.

The important idea is:

> **The individual number does not represent the position. The whole pattern of numbers identifies the position.**

For example:

```text
Position 0 → [0.00,  1.00,  0.00,  1.00]
Position 1 → [0.84,  0.54,  0.10,  0.99]
Position 2 → [0.91, -0.42, 0.20,  0.96]
```

The Transformer can distinguish position `0`, `1`, and `2` because their patterns are different.

---

### Why `sin` and `cos`?

We use mathematical waves to generate these patterns:

```python
if i % 2 == 0:
    pos_enc[pos, i] = np.sin(angle)
else:
    pos_enc[pos, i] = np.cos(angle)
```

So the dimensions alternate:

```text
Dimension:  0     1     2     3     4     5
            ↓     ↓     ↓     ↓     ↓     ↓
Function:  sin   cos   sin   cos   sin   cos
```

`sin` and `cos` produce smooth patterns that change as the position changes.

Different dimensions use different frequencies, which creates more distinctive patterns.

---

### What does `pos` and `i` mean?

```python
for pos in range(seq_len):
    for i in range(d_model):
```

* `pos` = **which position am I encoding?**
* `i` = **which dimension of the encoding am I calculating?**

For example:

```text
pos = 2
i = 3
```

means:

> Calculate the value for **position 2, dimension 3**.

---

### Why `i // 2`?

```python
2 * (i // 2)
```

`i // 2` groups dimensions into pairs:

```text
i:       0   1   2   3   4   5
i // 2:  0   0   1   1   2   2
```

So dimensions `0 & 1` use one frequency, `2 & 3` another, etc.

This gives different dimensions different wave patterns.

---

### The formula

```python
angle = pos / np.power(base, (2 * (i // 2)) / d_model)
```

The formula determines **how fast the sine/cosine wave changes** for each dimension.

You don't need to memorize the formula.

Remember the idea:

> **Different positions + different frequencies → different patterns of numbers.**

---

### Final step

The positional encoding is added to the token embedding:

```text
Token embedding
      +
Positional encoding
      ↓
Transformer input
```

So the Transformer receives both:

```text
Token embedding      → What is the token?
Positional encoding  → Where is the token?
```

### Key takeaway

**Positional encoding gives every position a unique numerical pattern so that the Transformer can understand the order of tokens.**

```text
Token meaning  +  Token position
       ↓
Transformer input
```
