# Basic operations on matrices


def shape(A):
    """Return (rows, cols) for a 2D list; raise if empty or ragged."""
    if not A or not isinstance(A[0], list):
        raise ValueError("Matrix must be a non-empty 2D list.")
    row_count = len(A)
    col_count = len(A[0])
    if any(len(row) != col_count for row in A):
        raise ValueError("Matrix rows must all have the same length.")
    return row_count, col_count


def addn(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same dimensions for addition.")
    rows, cols = shape(A)
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(A[i][j] + B[i][j])
        result.append(row)
    return result


def subn(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same dimensions for subtraction.")
    rows, cols = shape(A)
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(A[i][j] - B[i][j])
        result.append(row)
    return result


def scalmult(A, k):
    rows, cols = shape(A)
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(A[i][j] * k)
        result.append(row)
    return result

#transpose of matrixes
def trans(A):
    rows, cols = shape(A)
    C=[]
    for i in range(cols):
        rows = []
        for j in range(rows):
            rows.append(A[j][i])
        C.append(rows)
    return C

#vector multiplication
def vecmult(A, K):
    rows, cols = shape(A)
    if cols != len(K):
        raise ValueError("for multiplication number of columns in the matrix and the size of the vector must be equal")
    C=[]
    for i in range(rows):
        sum=0
        for j in range(cols):
            sum += A[i][j] *K[j]
            C.append(sum)

    return C

#now the most important
#matrix matrix multiplication

def matmull(A, B):
    rA, cA = shape(A)
    rB, cB = shape(B)
    if cA!=rB:
        raise ValueError("for multiplication number of columns in the first matrix and number of rows in the second must be equal")
    for i in range(rA):
        C = []
        rows= []
        for j in range(cB):
            sum = 0 
            for k in range(cA):
                sum += A[i][k] * B[k][j]
            rows.append(sum)
        C.append(rows)

        return C


#updated sample usage:
if __name__ == "__main__":
    A = [[1, 2, 3],
         [4, 5, 6]]
    B = [[7, 8, 9],
         [10, 11, 12]]

    print("Matrix A:")
    for row in A:
        print(row)

    print("\nMatrix B:")
    for row in B:
        print(row)

    print("\nA + B:")
    C = addn(A, B)
    for row in C:
        print(row)

#matrix subtraction
    print("\nA - B:")
    D = subn(A, B)
    for row in D:
        print(row)

#scalar multiplication
    k = 2
    print(f"\nA * {k}:")
    E = scalmult(A, k)
    for row in E:
        print(row)

    print("\nTranspose of A:")

#vector multiplication
    F = trans(A)
    for row in F:
        print(row)
    print("\nA * vector [1, 2, 3]:")
    vec = [1, 2, 3] 
    G = vecmult(A, vec)
    print(G)


#matrix multiplication
    print("\nA * B:")
    H = matmull(A, B)
    for row in H:
        print(row)
    print("\nShape of A:")
    print(shape(A))
