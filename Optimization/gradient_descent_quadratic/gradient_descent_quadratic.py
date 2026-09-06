# This function implements gradient descent for a quadratic function of the form f(x) = ax^2 + bx + c. It takes the coefficients a, b, and c, an initial guess x0, a learning rate lr, and the number of steps to perform. The function iteratively updates the value of x based on the gradient of the function and prints the current step, x value, and function value at each iteration. Finally, it returns the final value of x after the specified number of steps.


def gradient_descent_quadratic(
    a: float, b: float, c: float, x0: float, lr: float, steps: int
) -> float:
    """
    Perform gradient descent on a quadratic function f(x) = ax^2 + bx + c.

    Args:
        a (float): Coefficient of x^2.
        b (float): Coefficient of x.
        c (float): Constant term.
        x0 (float): Initial guess for x.
        lr (float): Learning rate.
        steps (int): Number of steps to perform.
    
    Returns:
        float: The final value of x after performing gradient descent.
    """

    x = x0

    for _ in range(steps):
        # Calculate the gradient of the quadratic function f(x) = ax^2 + bx + c
        gradient = 2 * a * x + b

        # Update x using the gradient descent formula
        x -= lr * gradient

        # Print the current step, x value, and function value
        print(f"Step {_ + 1}: x = {x}, f(x) = {a * x**2 + b * x + c}")

    return float(x)


# Example usage:
if __name__ == "__main__":
    a = 1.0
    b = -4.0
    c = 3.0
    x0 = 0.0
    steps = 50
    lr = 0.1

    final_x = gradient_descent_quadratic(a, b, c, x0, lr, steps)
