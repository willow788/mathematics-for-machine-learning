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
   

def grad(f, point, method= 'central', h=1e-5):
    pt = list(point)
    gradient = []
    n = len(pt)
    for i in range(n):
        pderivatives= partial_derivatives(f, pt, method= method, h=h, var_indx=i)
        gradient.append(pderivatives)
        return gradient
    


    #Example usage:
if __name__ == "__main__":
    import numpy as np
    
    #defining the function
    def f(x,y):
         return x**2 + y**3
    
    point = [1.0, 2.0]
    h = 1e-5
    
    gradient = grad(f, point, method='central', h=h)
    
    print("Gradient at point", point)
    print(gradient)
