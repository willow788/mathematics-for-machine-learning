import numpy as np
from torch import norm  

"""
this code computes the directinal derivative of a function f at point x
input : 
    f : function handle
    x : point at which the directional derivative is computed
    d : direction along which the directional derivative is computed
    
output :
    dir_deriv : directional derivative of f at x along d

"""
def directional_derivative(f, point, direction):
    # Normalize the direction
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)

    norm_val = np.linalg.norm(direction)
    if norm_val == 0:
        raise ValueError("Direction is supposed to be a non-zero vector")

    normalised_direction = direction / norm_val

    # Function to compute gradient numerically
    epsilon = 1e-8  # Small value for numerical differentiation
    gradient = np.zeros_like(point)
    for i in range(len(point)):
        dx = np.zeros_like(point)
        dx[i] = epsilon
        f_plus = f(point + dx)
        f_minus = f(point - dx)
        gradient[i] = (f_plus - f_minus) / (2 * epsilon)

    # Compute directional derivative
    dir_deriv = np.dot(gradient, normalised_direction)
    return dir_deriv


# Example usage
if __name__ == "__main__":
    def sample_function(x):
        return x[0]**2 + x[1]**2 + x[2]**2
    
    point = [1.0, 2.0, 3.0]
    direction = [4.0, 5.0, 6.0]
    result = directional_derivative(sample_function, point, direction)
    print("Directional Derivative:", result)
    print("point of directional derivative is:", point)
    print("direction of directional derivative is:", direction)


def directional_derivative_limit(f, point, direction, h=1e-5):
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)

    norm_val = np.linalg.norm(direction)
    if norm_val == 0:
        raise ValueError("Direction is supposed to be a non-zero vector")
    
    normalised_direction = direction / norm_val

    f_plus = f(point + h * normalised_direction)
    f_minus = f(point - h * normalised_direction)
    dir_deriv = (f_plus - f_minus) / (2 * h)
    return dir_deriv
