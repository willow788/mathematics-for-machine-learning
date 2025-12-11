#forward difference function
def forward_diff( f, point, var_indx, h):
    p = point.copy()
    p[var_indx]+=h
    return (f(*p) - f(*point))/h


#backward difference function
def back_diff(f, point, var_indx, h):
    p = point.copy()
    p[var_indx]-=h
    return (f(*point) - f(*p))/h


#central difference function
def cent_diff(f, point, var_indx, h):
    p1 = point.copy()
    p2 = point.copy()
    p1[var_indx]+=h
    p2[var_indx]-=h
    return (f(*p1) - f(*p2))/(2*h)

#wrapper function to choose method
def partial_derivatives(f, point, var_indx, h=1e-5, method='central'):
   if method == 'forward':
       return forward_diff(f, point, var_indx, h)
   
   elif method == 'backward':
       return back_diff(f, point, var_indx, h)
   elif method == 'central':
         return cent_diff(f, point, var_indx, h)
   else:
        raise ValueError("Method not recognized. Use 'forward', 'backward', or 'central'.")
   

   #Example usage:

if __name__ == "__main__":
    import numpy as np
    
    #defining the function
    def f(x,y):
         return x**2 + y**3
    
    point = [1.0, 2.0]
    h = 1e-5
    
    #computing partial derivative with respect to x (var_indx=0)
    pd_x_forward = partial_derivatives(f, point, 0, h, method='forward')
    pd_x_backward = partial_derivatives(f, point, 0, h, method='backward')
    pd_x_central = partial_derivatives(f, point, 0, h, method='central')
    
    print("Partial derivative with respect to x at point", point)
    print("Forward Difference:", pd_x_forward)
    print("Backward Difference:", pd_x_backward)
    print("Central Difference:", pd_x_central)
    
    #computing partial derivative with respect to y (var_indx=1)
    pd_y_forward = partial_derivatives(f, point, 1, h, method='forward')
    pd_y_backward = partial_derivatives(f, point, 1, h, method='backward')
    pd_y_central = partial_derivatives(f, point, 1, h, method='central')
    
    print("\nPartial derivative with respect to y at point", point)
    print("Forward Difference:", pd_y_forward)
    print("Backward Difference:", pd_y_backward)
    print("Central Difference:", pd_y_central)
