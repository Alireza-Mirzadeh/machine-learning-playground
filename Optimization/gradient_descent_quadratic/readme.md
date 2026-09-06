# Gradient Descent

## What is it?

**Gradient descent** is an optimization algorithm used to find the value of `x` that **minimizes a function**.

In simple words:

> Start somewhere, look at which direction is downhill, and take a small step in that direction.

For this problem, our function is a quadratic:

```text
f(x) = ax² + bx + c
```

For the example:

```python
a = 1.0
b = -4.0
c = 3.0
```

we get:

```text
f(x) = x² - 4x + 3
```

This function looks like a bowl:

```text
        f(x)
          ↑
          |
        \     /
         \   /
          \_/
           ↑
        minimum
```

Our goal is to find the `x` where the function reaches its minimum.

---

## 1. Start with an initial value

```python
x = x0
```

We need to choose a starting point.

In the example:

```python
x0 = 0.0
```

So we start at:

```text
x = 0
```

We are probably not at the minimum yet.

---

## 2. Calculate the gradient

```python
gradient = 2 * a * x + b
```

The **gradient** tells us the direction in which the function is increasing.

For:

```text
f(x) = ax² + bx + c
```

the derivative is:

```text
f'(x) = 2ax + b
```

The derivative/gradient tells us whether we should move `x` **left or right** to go downhill.

For example:

```text
gradient > 0  → move left
gradient < 0  → move right
gradient = 0  → at the minimum
```

Think of the gradient as telling us:

> **"Which way is downhill?"**

---

## 3. Update x

```python
x -= lr * gradient
```

This is the main gradient descent step.

It is equivalent to:

```python
x = x - lr * gradient
```

We move in the **opposite direction of the gradient** because the gradient points uphill.

* `gradient` → direction uphill
* `-gradient` → direction downhill
* `lr` → how big our step should be

---

## 4. Learning rate

```python
lr = 0.1
```

The learning rate controls the **size of each step**.

Think about walking down a hill:

```text
Small learning rate:

   \       /
    \ . . /
     \.../
      \_/
```

You take small steps.

A large learning rate takes bigger steps:

```text
   \       /
    \     /
     \   /
      \_/
```

If the learning rate is too small, learning can be very slow.

If it is too large, we can jump over the minimum and possibly move away from it.

---

## 5. Repeat the process

```python
for _ in range(steps):
```

We repeat:

```text
1. Calculate gradient
2. Move downhill
3. Repeat
```

The process looks like:

```text
Start
  ↓
Calculate gradient
  ↓
Update x
  ↓
Calculate gradient again
  ↓
Update x
  ↓
   ...
  ↓
Reach the minimum
```

---

## Example

Our function is:

```text
f(x) = x² - 4x + 3
```

We start at:

```text
x = 0
```

The gradient is:

```text
f'(x) = 2x - 4
```

### Step 1

At `x = 0`:

```text
gradient = 2(0) - 4
         = -4
```

The gradient is negative, so we need to move **right**.

Using:

```text
x = x - lr × gradient
```

with `lr = 0.1`:

```text
x = 0 - 0.1(-4)
  = 0.4
```

Now:

```text
x = 0.4
```

### Step 2

Calculate the gradient again:

```text
gradient = 2(0.4) - 4
         = -3.2
```

Update:

```text
x = 0.4 - 0.1(-3.2)
  = 0.72
```

Now we're closer to the minimum.

The process continues:

```text
0 → 0.4 → 0.72 → 0.976 → 1.1808 → ...
```

Eventually, `x` gets closer and closer to:

```text
x = 2
```

which is the minimum of this function.

---

## 6. Calculate the function value

```python
print(f"Step {_ + 1}: x = {x}, f(x) = {a * x**2 + b * x + c}")
```

This lets us see how the algorithm is progressing.

For our function:

```text
f(x) = x² - 4x + 3
```

the minimum occurs at:

```text
x = 2
```

and:

```text
f(2) = 4 - 8 + 3
     = -1
```

So gradient descent is trying to move toward:

```text
x ≈ 2
f(x) ≈ -1
```

---

## The complete idea

```text
              Start
                ↓
           x = x0
                ↓
       Calculate gradient
                ↓
       Which way is uphill?
                ↓
      Move in opposite direction
                ↓
          Update x
                ↓
            Repeat
                ↓
        x → minimum
```

### Key formula

```python
x = x - lr * gradient
```

Remember:

> **Gradient tells you which way is uphill. Gradient descent moves in the opposite direction.**

### Important concepts

| Concept     | Meaning                       |
| ----------- | ----------------------------- |
| `x0`        | Where we start                |
| `gradient`  | Direction of uphill           |
| `-gradient` | Direction of downhill         |
| `lr`        | Size of each step             |
| `steps`     | Number of updates             |
| `x`         | Current position              |
| `f(x)`      | Current value of the function |

### Key takeaway

**Gradient descent repeatedly calculates the gradient and moves `x` in the opposite direction until it gets close to the minimum of the function.**
