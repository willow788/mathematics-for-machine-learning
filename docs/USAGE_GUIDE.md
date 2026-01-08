# 🖥️ Usage Guide

## Running Examples

Make sure dependencies are installed via `pip install numpy matplotlib scipy`.

- **Gradients**  
  `$ python src/Multivariate\ Calculus/gradient.py`

- **Hessian matrix**  
  `$ python src/Multivariate\ Calculus/hessian.py`

- **Constrained optimization (Lagrange)**  
  `$ python src/Multivariate\ Calculus/lagrange\ multipliers/main.py`

- **3D function/contour plot**  
  `$ python src/linear\ algebra/visualising\ functions`

## Using the Functions in Code

```python
from gradient import grad
def f(x, y): return x**2 + 3*x*y + y**2
print(grad(f, [1.0, 2.0]))  # Output: [7.0, 5.0]
```

## Customization

- **Change mathematical functions:** Edit the relevant function definition directly.
- **Change grid or plotting parameters:** Adjust the mesh size, colormap, or range in the plotting scripts.
- **Add new visualizations:** Fork a plotting script and try out your own mathematical formula!

---

**For more examples and theory, see the README and `docs/ADVANCED_MATH.md`.**
