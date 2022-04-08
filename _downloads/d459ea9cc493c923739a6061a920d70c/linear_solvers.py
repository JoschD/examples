"""
.. _linear-solvers:

=======================
Linear Equation Solvers
=======================

A quick summary for linear equation solvers available in python packages.

"""
# %%
import numpy as np
import cvxpy as cp
import scipy
#%%
# Let's say we want to solve an equation system
#
# .. math::
#     A \cdot x = y

A = np.array([[1, 2], [4, 6]])
y = np.array([3, 6])
print("A=")
print(A)
print(f"{y=!s}")
# %%
# If the equation system is exact solvable, i.e. if the inverse of A exists,
# we can either calculate 
# 
# .. math::
#       x = A^{-1} \cdot y
# 
# .. warning::
#       Don't!

x = np.linalg.inv(A).dot(y)
print(f"{x=!s}")
# %%
# Or let numpy solve this for you:

x = np.linalg.solve(A, y)
print(f"{x=!s}")

# %%
# This does not work with non-invertable matrices!
A = np.array([[1, 2, 5], [4, 6, 1]])
y = np.array([3, 6])
try:
    x = np.linalg.solve(A, y)
except np.linalg.LinAlgError as e:
    print(f"1) {e.__class__.__name__}: {e!s}")


try:
        x = np.linalg.inv(A).dot(y)
except np.linalg.LinAlgError as e:
    print(f"2) {e.__class__.__name__}: {e!s}")
# %%
# But we can still find a solution for x
#
# via the linear least squares algorithm implemented in numpy:

x, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
print(f"{x=!s}")
print(f"{residuals=!s}")
print(f"{rank=!s}")
print(f"{sv=!s}")
print(f"A * x = {A @ x}")
# %%
# Or at least the x that fulfills: 
#
# .. math::
#       \min_{x} \left\lVert A \cdot x - y \right\rVert
#

A = np.array([[1, 1], [6, 1], [4, 6]])
y = np.array([3, 6, 8])
x, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
print(f"{x=!s}")
print(f"{residuals=!s}")
print(f"{rank=!s}")
print(f"{sv=!s}")
print(f"A * x = {A @ x}")

# %%
