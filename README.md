# Mathematics for Machine Learning — From First Principles

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

**Building mathematical intuition for machine learning through hands-on implementations**

[Getting Started](#-quick-start) • [Documentation](../../wiki) • [Examples](#-examples) • [Contributing](../../wiki/Contributing) ��� [FAQ](../../wiki/FAQ)

</div>

---

## 🎯 Overview

A collection of small, focused projects that build key mathematical concepts used in modern machine learning **from first principles**. Each project emphasizes hand-written derivations, minimal dependencies, and clear educational value.

### Why This Repository Exists

Modern ML frameworks often hide the underlying mathematics.  
This repository reveals how important concepts can be derived and implemented directly from definitions, helping you gain **intuition** and mathematical insight, not just code recipes.

**Perfect for:**
- 🎓 Students learning calculus, linear algebra, or ML fundamentals
- 💻 Developers wanting to understand what's under the hood
- 🔬 Researchers needing reference implementations
- 📚 Educators looking for teaching materials
- 🚀 Self-learners who prefer hands-on understanding

---

## ✨ What You'll Find Here

- **Hand-written mathematical derivations** - Clear explanations of theory
- **From-scratch implementations** - Using only basic numerical tools (minimal NumPy)
- **Rich visualizations** - 3D surfaces, contour plots, vector fields, transformations
- **Applied connections** - Linking pure math to ML optimization and algorithms
- **Educational focus** - Built to teach, not just compute

---

## 📂 Project Categories

### 🧮 Multivariate Calculus
*Status: ✅ Active*

| Implementation      | Description                          | Use Cases           |
|---------------------|--------------------------------------|---------------------|
| **Partial Derivatives**    | Forward, backward, and central difference methods | Sensitivity analysis  |
| **Gradient Computation**   | Numerical gradient using central differences      | Gradient descent, backpropagation |
| **Hessian Matrix**         | Second-order derivative matrix                   | Newton's method, convexity testing |
| **Lagrange Multipliers**   | Constrained optimization solver                  | SVMs, constrained neural networks |
| **Vector Field Visualization** | Streamline plots with particle flow          | Understanding dynamics, ODEs |

**Example outputs:**
- 3D surface plots showing optimization landscapes
- Contour maps with constraint circles
- Animated particle flows in vector fields

[📖 Multivariate Calculus Guide](../../wiki/Multivariate-Calculus)

---

### 🔢 Linear Algebra
*Status:  ✅ Active*

| Implementation           | Description                         | Planned Features              |
|--------------------------|-------------------------------------|-------------------------------|
| **Function Visualization**  | 3D surfaces + contour plots       | ✅ Complete                    |
| **Matrix Operations**       | Multiplication, inverse, determinant | 🔄 Upcoming                |
| **Eigenvalue Decomposition**| Power iteration, QR algorithm     | 🔄 Upcoming                    |
| **PCA from Scratch**        | Manual principal component analysis | 🔄 Upcoming                    |
| **Linear Transformations**  | Rotation, scaling, shearing       | 🔄 Upcoming                    |

**Current visualizations:**
- Dual-panel 3D surface and contour plots
- Rainbow colormaps for intuitive value mapping
- High-resolution grid rendering (1000×1000)

[📖 Linear Algebra Guide](../../wiki/Linear-Algebra)

---

### 🌊 Vector Calculus
*Status: 🔄 Planned*

- Divergence and curl computation
- Line and surface integrals
- Gradient, divergence, and curl visualizations
- Green's theorem, Stokes' theorem, divergence theorem

[📖 Vector Calculus Guide](../../wiki/Vector-Calculus)

---

### 🤖 Applied Machine Learning Math
*Status: 🔄 Planned*

- Gradient descent variants (SGD, momentum, Adam, RMSprop)
- Backpropagation for simple neural networks
- Loss landscape visualization
- Manual implementation of logistic regression
- Optimization trajectory analysis

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/willow788/mathematics-for-machine-learning.git
cd mathematics-for-machine-learning

# 2. Create virtual environment (recommended)
python -m venv .venv

# Activate on macOS/Linux: 
source .venv/bin/activate

# Activate on Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**If no requirements.txt exists:**
```bash
pip install numpy matplotlib scipy
```

### Running Examples

```bash
# Compute gradients
python src/Multivariate\ Calculus/gradient.py

# Compute Hessian matrix
python src/Multivariate\ Calculus/hessian.py

# Solve constrained optimization with Lagrange multipliers
python src/Multivariate\ Calculus/lagrange\ multipliers/main.py

# Visualize vector fields with particle flow
python src/Multivariate\ Calculus/vector\ visualisation/main.py

# Create 3D surface and contour plots
python src/linear\ algebra/visualising\ functions
```

---

## 💡 Examples

### Example 1: Computing a Gradient

```python
from gradient import grad

# Define function
def f(x, y):
    return x**2 + 3*x*y + y**2

# Compute gradient at point (1, 2)
point = [1.0, 2.0]
gradient = grad(f, point, method='central')
print(f"∇f({point}) = {gradient}")  
# Output: ∇f([1.0, 2.0]) = [7.0, 5.0]
```

**Interpretation:** At point (1, 2), the function increases fastest in direction [7, 5]. For minimization via gradient descent, move in direction [-7, -5].

---

### Example 2: Analyzing Curvature with Hessian

```python
from hessian import hessianFunc
import numpy as np

def f(x, y):
    return x**2 + y**2  # Simple bowl function

point = [1.0, 1.0]
H = hessianFunc(f, point)

print("Hessian matrix:")
for row in H:
    print(row)

# Analyze eigenvalues
eigenvalues = np.linalg.eigvals(np.array(H))
if all(eigenvalues > 0):
    print("✓ Convex function (local minimum)")
```

---

### Example 3: Constrained Optimization

```python
# Runs Lagrange multiplier optimization
# Minimizes:  f(x, y) = x² + 2y² - 2x
# Subject to: x² + y² = 4 (circle constraint)

python src/Multivariate\ Calculus/lagrange\ multipliers/main.py
```

**Output:**
- All critical points on the constraint circle
- Function values at each point
- Visualization showing contours + constraint + solutions

---

### Example 4: Visualizing Mathematical Functions

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))  # Radial wave pattern

# Creates side-by-side 3D surface and contour plots
python src/linear\ algebra/visualising\ functions
```

**Creates:**
- Left: 3D surface with rainbow colormap
- Right: Contour plot showing level curves
- Interactive rotation for 3D plot

---

## 🗂️ Repository Structure

```
mathematics-for-machine-learning/
├── src/
│   ├── Multivariate Calculus/
│   │   ├── PartialDerivatives.py
│   │   ├── gradient.py
│   │   ├── hessian.py
│   │   ├── vector visualisation/
│   │   │   └── main.py
│   │   └── lagrange multipliers/
│   │       ├── main.py
│   │       └── explanations.txt
│   ├── linear algebra/
│   │   └── visualising functions
│   ├── vector_calculus/        # (Planned)
│   └── applied_ml/             # (Planned)
├── notebooks/                  # Jupyter notebooks (planned)
├── tests/                      # Validation scripts (planned)
├── docs/                       # Extended documentation (planned)
├── README.md
├── LICENSE
└── requirements.txt
```

---

## 🎨 Example Visualizations

<div align="center">

### Multivariate Calculus

**Vector Field Flow** | **Lagrange Multipliers** | **3D Surface**
:--:|:--:|:--: 
Streamlines showing F(x,y)=(-y,x) | Constrained optimization visualization | Function landscape with contours

</div>

> **Note:** Sample visualizations coming soon! Run the scripts to see them yourself.

---

## 🎓 Learning Path

**Recommended progression:**

1. **Start with Partial Derivatives** → Understand how functions change
   ```bash
   python src/Multivariate\ Calculus/PartialDerivatives.py
   ```

2. **Move to Gradients** → Combine partials into optimization direction
   ```bash
   python src/Multivariate\ Calculus/gradient.py
   ```

3. **Understand Second-Order Info** → Curvature for faster optimization
   ```bash
   python src/Multivariate\ Calculus/hessian.py
   ```

4. **Explore Constrained Optimization** → Real-world constraints
   ```bash
   python src/Multivariate\ Calculus/lagrange\ multipliers/main.py
   ```

5. **Visualize Concepts** → Build geometric intuition
   ```bash
   python src/Multivariate\ Calculus/vector\ visualisation/main.py
   python src/linear\ algebra/visualising\ functions
   ```

---

## 🔗 Connections to Machine Learning

| Mathematical Concept    | ML Application              |
|------------------------|-----------------------------|
| **Gradient**           | Backpropagation, gradient descent |
| **Hessian**            | Second-order optimizers (Newton, L-BFGS) |
| **Partial Derivatives**| Chain rule, sensitivity analysis |
| **Lagrange Multipliers** | SVMs, constrained neural networks |
| **Vector Fields**      | Neural ODEs, dynamical systems |
| **Eigenvalues**        | PCA, spectral methods |
| **Matrix Decomposition** | Recommender systems, dimensionality reduction |
| **Contour Plots**      | Loss landscapes, decision boundaries |

---

## 🤝 Contributing

Contributions are **warmly welcomed**! Whether you're fixing typos, adding features, or improving documentation.

### Quick Start

1. **Fork** the repository
2. **Create a branch**:  `git checkout -b feat/amazing-feature`
3. **Make changes** with clear documentation
4. **Commit**: `git commit -m 'Add amazing feature'`
5. **Push**: `git push origin feat/amazing-feature`
6. **Open a Pull Request**

### Contribution Guidelines

- **Minimal dependencies** - Prefer NumPy over high-level ML libraries
- **Clear documentation** - Explain the math, not just the code
- **Visual examples** - Include plots and visualizations
- **Mathematical rigor** - Verify correctness against known solutions

**See the full [Contributing Guide](../../wiki/Contributing) for details.**

---

## 📚 Documentation

Comprehensive guides available in the [Wiki](../../wiki):

- **[Home](../../wiki/Home)** - Repository overview and navigation
- **[Getting Started](../../wiki/Getting-Started)** - Detailed setup instructions
- **[Multivariate Calculus](../../wiki/Multivariate-Calculus)** - In-depth guide to calculus implementations
- **[Linear Algebra](../../wiki/Linear-Algebra)** - Matrix operations and visualizations
- **[Vector Calculus](../../wiki/Vector-Calculus)** - Planned content overview
- **[Contributing](../../wiki/Contributing)** - How to contribute
- **[FAQ](../../wiki/FAQ)** - Frequently asked questions

---

## 📖 Learning Resources

### Books
- *Calculus* by James Stewart
- *Linear Algebra Done Right* by Sheldon Axler
- *Convex Optimization* by Boyd & Vandenberghe
- *Deep Learning* by Goodfellow, Bengio, and Courville

### Online Courses
- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown - Multivariable Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [MIT OCW 18.06 - Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [Stanford CS229 - Machine Learning](http://cs229.stanford.edu/)

---

## 🐞 Issues and Support

- **Bug reports**:  [Open an issue](../../issues/new?template=bug_report.md)
- **Feature requests**: [Open an issue](../../issues/new?template=feature_request.md)
- **Questions**: Check the [FAQ](../../wiki/FAQ) or [start a discussion](../../discussions)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use in research and education
- ✅ Include in your projects

Just include the license and give attribution! 🙏

---

## 🌟 Acknowledgements

This repository is inspired by the desire to build mathematical intuition for ML by: 
- Re-deriving concepts from first principles
- Visualizing abstract mathematics
- Avoiding black-box libraries
- Emphasizing understanding over application

**Special thanks to:**
- The open-source community
- Contributors who help improve this resource
- Educators who make mathematics accessible
- Students who inspire continuous learning

---

<div align="center">

## 🚀 Roadmap

### Current Focus
- ✅ Multivariate calculus implementations
- ✅ Basic visualizations
- ✅ Documentation and wiki pages

### Next Steps
- 🔄 Complete linear algebra implementations
- 🔄 Add Jupyter notebooks for interactive learning
- 🔄 Create video tutorials
- 🔄 Implement vector calculus modules
- 🔄 Build applied ML examples

### Future Plans
- 📅 Neural network backpropagation from scratch
- 📅 Optimization algorithm comparisons
- 📅 Interactive web-based visualizations
- 📅 Community-contributed projects

---

**⭐ If you find this repository helpful, please consider giving it a star!  ⭐**

**Made with ❤️ for the ML learning community**

[⬆ Back to Top](#mathematics-for-machine-learning--from-first-principles)

</div>
