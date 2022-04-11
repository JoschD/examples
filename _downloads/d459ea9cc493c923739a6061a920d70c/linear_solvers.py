"""
.. _linear-solvers:



=======================
Linear Equation Solvers
=======================

A quick summary for linear equation solvers available in python packages.

"""
# %%
# Dependencies:

import numpy as np
import cvxpy as cp
import scipy

#%%
# Simple System of Linear Equations
# =================================
#
# Let's say we want to solve an equation system
#
# .. math::
#     A \cdot x = y
#
#
# Inverse 
# -------
# If the equation system is exact solvable, i.e. if the inverse of A exists,
# we can easily calculate 
# 
# .. math::
#       x = A^{-1} \cdot y
# 
# from numpy's linalg functions.
#

# Generate data.
A = np.array([[1, 2], 
              [4, 6]])
y = np.array([3, 6])

# Solve.
x = np.linalg.inv(A).dot(y)

# Print result.
print(f"{x = !s}")

# %%
# .. warning::
#       | **Don't!**
#       | Don't ever do that! 
#
#       Inverting a matrix is numerically much more unstable than 
#       using proper solvers for the whole equation system.
#
# Solve
# -----
# It is better to let numpy solve this for you:

# Generate data.
A = np.array([[1, 2], 
              [4, 6]])
y = np.array([3, 6])

# Solve.
x = np.linalg.solve(A, y)

# Print result.
print(f"{x = !s}")

# %%
# But, this only works on quadratic matrices.
# It fails if the matrix does not have an inverse:
#

# Generate data.
A = np.array([[1, 2, 5], 
              [4, 6, 1]])
y = np.array([3, 6])

# 1) Solve via inverse.
try:
    x = np.linalg.inv(A).dot(y)
except np.linalg.LinAlgError as e:
    print(f"1) {e.__class__.__name__}: {e!s}")

# 3) Solve with `solve` function.
try:
    x = np.linalg.solve(A, y)
except np.linalg.LinAlgError as e:
    print(f"2) {e.__class__.__name__}: {e!s}")

# %%
# Least Squares
# -------------
# Yet, we can still find a solution for x
# via the linear least squares algorithm implemented in numpy.
# It finds the x that minimizes the norm of the residual: 
#
# .. math::
#       \min_{x} \left\lVert A \cdot x - y \right\rVert
#
# In *underdetermined* systems it finds an *exact* solution:
#

# Generate data.
A = np.array([[1, 2, 5], 
              [4, 6, 1]])
y = np.array([3, 6])

# Solve.
x, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)

# Print result.
print(f"{x = !s}")
print(f"{A @ x = !s}")
print(f"{residuals = !s}")
print(f"{rank = !s}  (of A)")
print(f"{sv = !s} (singular values of A)")

# %%
# And the same function optimizes the residuals for
# an *overdetermined* equation system. 
#

# Generate data.
A = np.array([[1, 1], 
              [6, 1], 
              [4, 6]])
y = np.array([3, 6, 8])

# Solve.
x, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)

# Print result.
print(f"{x = !s}")
print(f"{A @ x = !s}")
print(f"{residuals  = !s}")
print(f"{rank = !s}  (of A)")
print(f"{sv = !s} (singular values of A)")

# %%
# Linear Equations with constraints
# =================================
#
#  
# .. math::
#     A \cdot x = y
# 
# such that
#  
# .. math::
#    B \cdot x \leq z
#

# %%
# If you need an exact solution for your equalities,
# one can abuse `scipy`'s `linprog`, which finds 
# a solution to the problem
#
# .. math::
#    \min_{x} c^T x
#
# such that
# 
# .. math::
#    A_{ub} \cdot x \leq b_{ub} \\
#    A_{eq} \cdot x = b_{eq} \\
#      l \leq x \leq u 
#
# Where we can set :math:`c` to zero, as we don't care about the minimization.
# Using the linear least-squares example above, we had one value > 0.6.
# Let us force all values of x to be below 0.6. 
# We only need to plug-in our equality system (eq) and the bounds:
#

# Generate data.
A = np.array([[1, 2, 5], 
              [4, 6, 1]])
y = np.array([3, 6])
c = np.zeros(A.shape[1])

# Solve.
result = scipy.optimize.linprog(c=c, A_ub=None, b_ub=None, A_eq=A, b_eq=y, bounds=[[None, 0.6]]*3)
x = result.x

# Print result.
print(result.message)
print(f"{x = !s}")
print(f"{A @ x = !s}")
print(f"{result.con = !s} (i.e. residuals)")

# %%
# Alternatively, or if our bounds come from another equation system, 
# we can give upper-bounds system (ub) instead of the `bounds`:
# (In this case, this leads to better residuals!)

# Generate data.
A = np.array([[1, 2, 5], 
              [4, 6, 1]])
y = np.array([3, 6])
B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
z = 0.6 * np.ones(B.shape[1])
c = np.zeros(A.shape[1])

# Solve.
result = scipy.optimize.linprog(c=c, A_ub=B, b_ub=z, A_eq=A, b_eq=y, bounds=None)
x = result.x

# Print result.
print(result.message)
print(f"{x = !s}")
print(f"{A @ x = !s}")
print(f"{result.slack = !s} (i.e. z - x)")
print(f"{result.con = !s} (i.e. residuals)")

# %%
# However, this does **not** minimize 

# Generate data.
m = 20
n = 15
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
cost = cp.sum_squares(A @ x - b)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)
print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)

# %%
# Nonlinear Optimization
# ----------------------
# 
# If the problem is non-linear or with error-bars, there are a lot of functions available
#
# - | `scipy.optimize.least_squares(fun, x0, jac='2-point', bounds=(- inf, inf)) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares>`_ 
#   | Solve a nonlinear least-squares problem with bounds on the variables.
#
# - | `scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(- inf, inf)) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit>`_ 
#   | Use non-linear least squares to fit a function, f, to data. *(Basically a wrapper around least_squares)*
#
# - | `scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='interior-point', callback=None, options=None, x0=None) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html>`_ 
#   | Linear programming: minimize a linear objective function subject to linear equality and inequality constraints.
#   
# 


# %%
# Thumbnail for the Sphinx-Gallery:
# sphinx_gallery_thumbnail_path = '_static/thumb_linear_solvers.png'