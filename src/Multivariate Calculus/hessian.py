def cent_diff(f, point, var_indx, h):
    p1 = point.copy()
    p2 = point.copy()
    p1[var_indx] += h
    p2[var_indx] -= h
    return (f(*p1) - f(*p2)) / (2 * h)

def part_derivative(f, point, var_indx, h=1e-5):
    return cent_diff(f, point, var_indx, h)

def second_deriv(f, point, var_i, var_j, h= 1e-5):
    p1 = point.copy()
    p2 = point.copy()
    p1[var_j] += h
    p2[var_j] -= h

    d1 = part_derivative(f, p1, var_i, h)
    d2 = part_derivative(f, p2, var_i, h)

    return(d1-d2)/2*h

def hessianFunc(f, point, h=1e-5):
    n = len(point)
    H=[]
    for i in range(n):
        row = []
        for j in range(n):
            row.append(second_deriv(f, point, i, j, h))
            H.append(row)
    return H

    #Example usage:
if __name__ == "__main__":
    import numpy as np

    #defining the function
    def f(x, y):
        return x**2 + y**3

    point = [1.0, 2.0]
    h = 1e-5

    hessian_matrix = hessianFunc(f, point, h=h)

    print("Hessian matrix at point", point)
    for row in hessian_matrix:
        print(row)
