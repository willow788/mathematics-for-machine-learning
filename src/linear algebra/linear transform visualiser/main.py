import matplotlib.pyplot as plt

def matvec(A, x):
    """multiplying a 2x2 matrix A by a 2D vector x"""
    return[
        A[0][0]* x[0] + A[0][1]* x[1],
        A[1][0]* x[0] + A[1][1]* x[1]
    ]

#generatig grid points;
def gen_grid(n):
    """generating the grid of points from -n to n"""
    points = []
    for x in range(-n, n+1):
        for y in range(-n, n+1):
            points.append([x,y])

            return points
        
#applying linear transformation to grid points
def apply_trans(A, points):
    """applying linear transformation in matrix A to the list of points"""

    return [matvec(A, p) for p in points]

#visualising the plot points

def plot_points(original, transformed):

    ox = [p[0] for p in original]
    oy = [p[1] for p in original]   

    tx = [p[0] for p in transformed]
    ty = [p[1] for p in transformed]

    plt.figure(figsize=(8,8))

    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')

    plt.scatter(ox, oy, color='blue', label='Original Points')

    plt.scatter(tx, ty, color='red', label='Transformed Points')

    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.title("Linear Transformation Visualiser")
    plt.show()

#example usage
if __name__ == "__main__":
    A = [[2, 1],
         [1, 3]]

    n = 5
    original_points = gen_grid(n)
    transformed_points = apply_trans(A, original_points)
    plot_points(original_points, transformed_points)
    

