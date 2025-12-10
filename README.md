```markdown
# Mathematics for Machine Learning — From First Principles

A collection of small, focused projects that build key mathematical concepts used in modern machine learning from first principles. Each project emphasizes hand-written derivations, minimal dependencies, and visual intuition through plots and interactive examples.

Why this repository exists
- Modern ML frameworks often hide the underlying math. This repo shows how important concepts can be derived and implemented directly from definitions, so you gain intuition rather than only applying black-box tools.
- Ideal for learners who want to deeply understand multivariate calculus, linear algebra, vector calculus, and the mathematical building blocks of ML algorithms.

What you'll find here
- Hand-written mathematical derivations and explanations.
- Code implementations that use only basic numerical tools (minimal NumPy / Python; many examples avoid high-level ML abstractions).
- Visualizations (3D surfaces, contour plots, vector fields, transformations) to build geometric intuition.
- Small applied projects that connect the math to ML ideas (optimization, loss landscapes, manual backpropagation).

Project categories
- Multivariate Calculus
  - Gradients, Jacobians, Hessians
  - 3D surfaces, contour maps, optimization landscapes (saddle points, minima, maxima)
- Linear Algebra
  - Matrix arithmetic implemented from scratch
  - Eigenvalues & eigenvectors, diagonalization
  - PCA implemented manually, linear transformations visualized
  - Orthogonality, projections, change of basis
- Vector Calculus (planned)
  - Divergence, curl, fields, line & surface integrals
  - Visual and computational intuition for PDE-related operators
- Applied Machine Learning Math (planned)
  - Gradient descent and variants implemented directly
  - Logistic regression, simple neural nets and backpropagation built from scratch
  - Visualizing loss geometry and optimization behavior

Example outputs
- 3D surface plots and contour maps produced by the projects.
- Plots demonstrate how analytic expressions translate into geometric behavior and optimization trajectories.

Quick start
1. Clone the repo:
   git clone https://github.com/willow788/mathematics-for-machine-learning.git
   cd mathematics-for-machine-learning

2. (Optional) Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows

3. Install dependencies (if the project includes a requirements file):
   pip install -r requirements.txt

4. Run a script or notebook:
   - Python script: python path/to/project_script.py
   - Jupyter notebook: jupyter lab or jupyter notebook and open the notebook file

Repository layout (typical)
- multivariate_calculus/   — functions, plots, examples
- linear_algebra/         — matrix code, eigen demos, PCA
- vector_calculus/        — (upcoming) fields and integrals
- applied_ml/             — (upcoming) optimization & small ML builds
- notebooks/              — interactive explorations and visualizations
- utils/                  — helper functions used across projects

Contributing
- Contributions are welcome. Suggested workflow:
  1. Open an issue describing the feature, bug, or project idea.
  2. Create a branch named `feat/...` or `fix/...`.
  3. Submit a pull request with clear description, examples, and any generated figures / notebooks.
- Aim for minimal dependencies, clear math derivations, and visual examples.

License
- Check the repository's LICENSE file for licensing details.

Support / Contact
- Open an issue on this repository for questions, suggestions, or to propose new projects.

Acknowledgements
- This repository is inspired by the desire to build mathematical intuition for ML by re-deriving and visualizing core ideas rather than relying solely on high-level libraries.
```
