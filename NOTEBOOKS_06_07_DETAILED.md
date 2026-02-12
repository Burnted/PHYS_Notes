# PHYSICS 234: DETAILED SUMMARIES FOR NOTEBOOKS 06 & 07

## 06 Solving, Minimizing, Fitting.ipynb - COMPREHENSIVE GUIDE

### Topics Covered
**Primary Focus: Solving Equations and Fitting Data to Models**

Advanced numerical techniques for optimization and parameter estimation.

### Key Concepts in Detail

1. **Root Finding Fundamentals**
   - **Definition**: Find x where f(x) = 0
   - **Multiple methods exist**: bisection (slow, reliable), Newton-Raphson (fast, needs derivative), secant (no derivative)
   - **System solving**: Can solve multiple equations simultaneously

2. **Function Minimization**
   - **Goal**: Find x that minimizes f(x)
   - **Related to root finding**: Minimize |f(x)| to solve f(x) = 0
   - **Optimization landscape**: Multiple local minima possible

3. **Parameter Fitting to Data**
   - **Least-squares objective**: Chi-squared = $\sum_i (data_i - model_i)^2 / \sigma_i^2$
   - **Weighted vs unweighted**: Weight by measurement uncertainty
   - **Covariance matrix**: Shows parameter correlations and uncertainties

### Important Root-Finding Methods Available

The notebook demonstrates several methods via `scipy.optimize.root_scalar()`:

| Method | Speed | Accuracy | Derivative? | Bracket? | Best For |
|--------|-------|----------|-------------|----------|----------|
| **bisect** | Slow | Good | No | Yes | Reliable, no assumptions |
| **brentq/brenth** | Medium | Excellent | No | Yes | Default choice |
| **newton** | Fast | Good | Yes | No | When derivative available |
| **secant** | Fast | Good | No | No | When derivative unknown |
| **toms748** | Fast | Excellent | No | Yes | Difficult functions |
| **halley** | Fast | Excellent | Yes (2nd) | No | High precision needed |

### Important Functions & Examples

#### 1. **scipy.optimize.root() - System of Equations**
```python
from scipy.optimize import root
import numpy as np

# System of 2 equations
def equations(vars):
    x, y = vars
    eq1 = x**2 + y**2 - 1      # Circle: x^2 + y^2 = 1
    eq2 = x - y                 # Line: x = y
    return [eq1, eq2]

# Initial guess
x0 = [1.0, 1.0]

# Solve system
solution = root(equations, x0, method='hybr')
print(solution.x)      # Solution
print(solution.success) # Did it converge?
print(solution.message) # Status message
```
**How to Use**: Solve f₁(x,y) = 0, f₂(x,y) = 0 simultaneously
**Key Points**:
- Returns OptimizeResult object with x (solution), success, message, nfev (function evaluations)
- Try method='hybr' first (hybrid Powell method)
- Can also use 'lm' (Levenberg-Marquardt), 'broyden1', 'broyden2'

#### 2. **scipy.optimize.root_scalar() - Single Equation**
```python
from scipy.optimize import root_scalar

def f(x):
    return x**3 - 2  # Solve x^3 = 2

# Method 1: Bisection (needs bracket)
solution = root_scalar(f, bracket=[1, 2], method='bisect')
print(f"Root: {solution.root}")  # x = 2^(1/3) ≈ 1.26

# Method 2: Newton-Raphson (needs derivative)
def f_prime(x):
    return 3*x**2

solution = root_scalar(f, x0=1.5, fprime=f_prime, method='newton')

# Method 3: Brent (hybrid, usually best)
solution = root_scalar(f, bracket=[1, 2], method='brentq')
```
**How to Use**: Find single root efficiently
**Key Points**:
- bracket=[a, b]: requires f(a) and f(b) have opposite signs
- x0: initial guess for derivative-based methods
- fprime: derivative function (optional)
- 'brentq' is usually the most robust choice

#### 3. **scipy.optimize.minimize() - Find Minimum**
```python
from scipy.optimize import minimize
import numpy as np

# Function to minimize
def objective(x):
    return (x[0] - 3)**2 + (x[1] - 2)**2 + np.sin(x[0])

# Initial guess
x0 = [0, 0]

# Simple minimization
result = minimize(objective, x0, method='Nelder-Mead')
print(f"Minimum at: {result.x}")
print(f"Function value: {result.fun}")
print(f"Iterations: {result.nit}")
print(f"Function evals: {result.nfev}")

# With gradient (for speed)
def gradient(x):
    return np.array([2*(x[0]-3) + np.cos(x[0]), 2*(x[1]-2)])

result = minimize(objective, x0, jac=gradient, method='BFGS')
```
**How to Use**: Minimize objective function
**Key Points**:
- method='Nelder-Mead': No gradient needed, slower but robust
- method='BFGS': Uses gradient estimate, faster
- method='L-BFGS-B': Bounded optimization
- jac: gradient function (optional, computed numerically if not provided)
- result.x = location of minimum, result.fun = value at minimum

#### 4. **scipy.optimize.curve_fit() - Fit Data to Model**
```python
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Generate noisy data
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2.1, 4.1, 5.8, 8.2, 9.9]) + np.random.normal(0, 0.2, 5)

# Model function with parameters p0, p1, p2
def model(x, p0, p1, p2):
    return p0 + p1*x + p2*x**2

# Initial guess for parameters
p_init = [1, 1, 0.1]

# Fit model to data
params, covariance = curve_fit(model, x_data, y_data, p0=p_init)
p0_fit, p1_fit, p2_fit = params

# Calculate parameter uncertainties
uncertainties = np.sqrt(np.diag(covariance))
print(f"p0 = {p0_fit:.3f} ± {uncertainties[0]:.3f}")
print(f"p1 = {p1_fit:.3f} ± {uncertainties[1]:.3f}")
print(f"p2 = {p2_fit:.3f} ± {uncertainties[2]:.3f}")

# Evaluate fitted model
x_fine = np.linspace(1, 5, 100)
y_fit = model(x_fine, *params)

# Plot
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_fine, y_fit, '-', label='Fit')
plt.legend()
```
**How to Use**: Fit model parameters to experimental data
**Key Points**:
- Returns (params, covariance) tuple
- covariance diagonal gives parameter variances, off-diagonal shows correlations
- p0: initial parameter guess (helps convergence)
- sigma: measurement uncertainties (weighted least squares)
- Can include maxfev to limit function evaluations

#### 5. **Least-Squares Fitting via Minimization**
```python
from scipy.optimize import minimize
import numpy as np

# Data and model
x_data = np.linspace(0, 10, 20)
y_data = 2 + 3*x_data + 0.1*x_data**2 + np.random.normal(0, 0.5, 20)

def model(params, x):
    p0, p1, p2 = params
    return p0 + p1*x + p2*x**2

# Chi-squared objective (what we minimize)
def chi_squared(params):
    y_pred = model(params, x_data)
    residuals = y_data - y_pred
    return np.sum(residuals**2)  # Or divide by sigma^2 for weighted fit

# Minimize chi-squared
result = minimize(chi_squared, x0=[0, 0, 0], method='Nelder-Mead')
best_params = result.x

print(f"Best fit parameters: {best_params}")
print(f"Chi-squared: {result.fun}")
```
**How to Use**: Alternative to curve_fit using minimize directly
**Key Points**:
- More flexible than curve_fit
- Can add additional constraints
- Can use different error models (L1, Huber, etc.)

---

## 07 Interpolation.ipynb - COMPREHENSIVE GUIDE

### Topics Covered
**Primary Focus: Estimating Function Values Between Data Points**

This notebook thoroughly explores different interpolation techniques, from simple linear to sophisticated splines.

### Key Concepts in Detail

1. **Why Interpolation?**
   - Extract smooth function from discrete data points
   - Evaluate function at intermediate points not in dataset
   - Calculate derivatives/integrals of tabulated data
   - Fill gaps in measurements

2. **Linear Interpolation**
   - **Method**: Straight line between consecutive points
   - **Continuity**: Function continuous, but derivative has jumps
   - **Advantages**: Simple, fast, always monotonic between points
   - **Disadvantages**: Not smooth, unrealistic for smooth physical functions
   - **Error**: O(h²) where h is spacing

3. **Cubic Spline Interpolation**
   - **Key property**: Second derivative is continuous
   - **Method**: Use cubic polynomials between data points
   - **Unknowns**: Value and second derivative at each point
   - **Result**: Smooth curve through all points
   - **Advantages**: Realistic for smooth functions, better accuracy
   - **Error**: O(h⁴) where h is spacing
   - **Disadvantage**: Interpolated value depends on distant points (global effect)

4. **Polynomial Interpolation (Advanced)**
   - **Barycentric Interpolation**: Numerically stable polynomial fitting
   - **Efficient re-evaluation**: Can change y-values without recalculating weights
   - **Higher accuracy** possible but with caveats

5. **Extrapolation Dangers**
   - **Inside data range**: Interpolation is generally safe
   - **Outside data range**: Catastrophic failures possible!
   - **Polynomial extrapolation**: Can oscillate wildly
   - **Physical violations**: May violate constraints (e.g., negatives when should be positive)
   - **Rule**: Never extrapolate beyond ~1 point spacing outside data

### Test Functions Used in Notebook

**Smooth Function**:
$$f_1(x) = \sin(x) + 0.1x$$
- Well-behaved, smooth, easily interpolated

**Oscillatory Function**:
$$f_2(x) = \sin(10x) \cdot e^{-x/2}$$
- Rapidly oscillating, decaying
- Demonstrates interpolation challenges with oscillations

### Important Functions & Examples

#### 1. **scipy.interpolate.interp1d() - General 1D Interpolation**
```python
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

# Data points
x_data = np.array([0, 1, 2, 3, 4, 5])
y_data = np.array([0, 0.8, 0.9, 0.1, -0.8, -0.3])

# Create interpolators of different types
f_linear = interp1d(x_data, y_data, kind='linear')
f_cubic = interp1d(x_data, y_data, kind='cubic')
f_quadratic = interp1d(x_data, y_data, kind='quadratic')

# Evaluate at finer grid
x_fine = np.linspace(0, 5, 100)
y_linear = f_linear(x_fine)
y_cubic = f_cubic(x_fine)

# Plot comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x_data, y_data, 'ro', label='Data')
plt.plot(x_fine, y_linear, 'b-', label='Linear')
plt.title('Linear Interpolation')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x_data, y_data, 'ro', label='Data')
plt.plot(x_fine, y_cubic, 'g-', label='Cubic')
plt.title('Cubic Spline')
plt.legend()

# Error comparison (if true function known)
f_true = lambda x: np.sin(x)
error_linear = np.abs(y_linear - f_true(x_fine))
error_cubic = np.abs(y_cubic - f_true(x_fine))

plt.subplot(1, 3, 3)
plt.plot(x_fine, error_linear, 'b-', label='Linear')
plt.plot(x_fine, error_cubic, 'g-', label='Cubic')
plt.yscale('log')
plt.title('Interpolation Error')
plt.legend()

plt.tight_layout()
plt.show()
```
**How to Use**: Choose interpolation method based on smoothness needed
**Key Points**:
- kind='linear': Fast, piecewise linear
- kind='cubic': Smooth derivatives, usually best
- kind='quadratic': In between (rarely used)
- fill_value='extrapolate': Extends outside range (WARNING: unsafe!)
- Returns callable function: y = f(x_new)

#### 2. **scipy.interpolate.make_interp_spline() - Cubic Spline with Control**
```python
from scipy.interpolate import make_interp_spline
import numpy as np

# Data with different spacing
x = np.array([0, 0.5, 1.0, 1.5, 2.0])
y = np.sin(2*np.pi*x)

# Create spline object (returns BSpline)
spl = make_interp_spline(x, y, k=3)  # k=3 for cubic spline

# Evaluate spline at many points
x_fine = np.linspace(0, 2, 200)
y_spline = spl(x_fine)

# Get derivatives
dy_spline = spl(x_fine, 1)  # First derivative
d2y_spline = spl(x_fine, 2)  # Second derivative

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x, y, 'ro', label='Data')
plt.plot(x_fine, y_spline, 'b-', label='Spline')
plt.title('Function')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x_fine, dy_spline, 'g-')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('First Derivative')

plt.subplot(3, 1, 3)
plt.plot(x_fine, d2y_spline, 'r-')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('Second Derivative (Continuous!)')

plt.tight_layout()
```
**How to Use**: Direct control over spline properties
**Key Points**:
- Returns BSpline object with derivative capability
- spl(x, n): Evaluate derivative of order n
- k=3: cubic (k=1 linear, k=2 quadratic, k=3 cubic)
- Better for integration/differentiation than interp1d

#### 3. **scipy.interpolate.BarycentricInterpolator() - Polynomial Interpolation**
```python
from scipy.interpolate import BarycentricInterpolator
import numpy as np

# Data points (should be well-spaced)
x = np.linspace(-5, 5, 10)  # Chebyshev spacing preferred for polynomials
y = np.sin(x)

# Create barycentric interpolator
P = BarycentricInterpolator(x, y)

# Evaluate at new points
x_fine = np.linspace(-5, 5, 200)
y_poly = P(x_fine)

# IMPORTANT: Can change y-values without recalculating weights!
y_new = np.cos(x)
P.set_y(y_new)  # Update y-values
y_poly2 = P(x_fine)

# Get derivatives
dy_poly = P(x_fine, der=1)

plt.plot(x, y, 'ro', label='Original data')
plt.plot(x_fine, y_poly, 'b-', label='Polynomial interpolation')
plt.legend()
```
**How to Use**: Numerically stable polynomial fitting
**Key Points**:
- Much better than fitting power series directly
- set_y(): Efficiently change y-values
- der=1,2,3: Compute derivatives
- Watch out: Polynomials can oscillate wildly near edges (Runge phenomenon)
- Best with Chebyshev spacing rather than uniform spacing

#### 4. **scipy.interpolate.UnivariateSpline() - Spline Smoothing for Noisy Data**
```python
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt

# Generate noisy data
x = np.linspace(0, 10, 100)
y_true = np.sin(x) * np.exp(-x/10)
y_noisy = y_true + np.random.normal(0, 0.1, len(x))

# Create smoothing splines with different smoothness
s_values = [0.001, 0.1, 1.0, 10]  # Higher s = more smoothing
splines = [UnivariateSpline(x, y_noisy, s=s) for s in s_values]

# Evaluate and plot
x_fine = np.linspace(0, 10, 500)
plt.figure(figsize=(14, 5))

for i, (s, spline) in enumerate(zip(s_values, splines)):
    plt.subplot(1, 4, i+1)
    plt.plot(x, y_noisy, 'o', alpha=0.5, label='Noisy data')
    plt.plot(x, y_true, 'r--', label='True function')
    plt.plot(x_fine, spline(x_fine), 'b-', label='Spline')
    plt.title(f's = {s}')
    plt.legend()

plt.tight_layout()

# Can integrate and differentiate spline
spline = UnivariateSpline(x, y_noisy, s=1.0)

# Integrate from 0 to 5
integral = spline.integral(0, 5)
print(f"Integral from 0 to 5: {integral:.4f}")

# Get smoothed y and derivatives
y_smooth = spline(x)
dy = spline(x, 1)  # First derivative
d2y = spline(x, 2)  # Second derivative

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(x, y_noisy, 'o', alpha=0.3)
plt.plot(x, y_smooth, '-')
plt.title('Smoothed function')

plt.subplot(3, 1, 2)
plt.plot(x, dy, '-')
plt.title('First derivative')

plt.subplot(3, 1, 3)
plt.plot(x, d2y, '-')
plt.title('Second derivative')

plt.tight_layout()
```
**How to Use**: Smooth noisy data with automatic smoothness selection
**Key Points**:
- s: Smoothing parameter
  - s << 1: Interpolates data points exactly (s=0)
  - s >> 1: Very smooth, approaches least squares fit
  - s ≈ # of data points: Often good default
- Can call with array of x values or single point
- der=0,1,2: Get function value, 1st derivative, 2nd derivative

#### 5. **scipy.interpolate.RectBivariateSpline() - 2D Interpolation**
```python
from scipy.interpolate import RectBivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create 2D data on regular grid
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 2, 3])

# Function values on grid
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create 2D spline interpolator
spline_2d = RectBivariateSpline(x, y, Z)

# Evaluate at finer grid
x_fine = np.linspace(0, 4, 50)
y_fine = np.linspace(0, 3, 50)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

# Evaluate spline (note: returns 2D array if grid=True)
Z_interp = spline_2d(x_fine, y_fine, grid=True)

# Can also get derivatives
# dZ_dx = spline_2d(x_fine, y_fine, dx=1, dy=0, grid=True)
# dZ_dy = spline_2d(x_fine, y_fine, dx=0, dy=1, grid=True)

# Plot
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5, label='Data')
ax.plot_surface(X_fine, Y_fine, Z_interp, alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('2D Spline Interpolation')

plt.show()
```
**How to Use**: Interpolate functions of two variables
**Key Points**:
- Requires data on regular grid (rectangular)
- grid=True: Returns 2D array for meshgrid
- grid=False: Returns 1D array for individual points
- dx, dy: Take partial derivatives

### Comparison of Interpolation Methods

| Method | Speed | Smoothness | Memory | Best Use | Worst Case |
|--------|-------|-----------|--------|----------|------------|
| **Linear** | Fast | C⁰ | Low | Quick plots, discrete data | Oscillations look linear |
| **Cubic spline** | Medium | C² | Medium | General purpose | Overshooting on noisy data |
| **Polynomial** | Slow | C∞ | Medium | Smooth functions, small N | Runge oscillations near edges |
| **Barycentric poly** | Slow | C∞ | Medium | Accurate evaluation | Still oscillates at edges |
| **Smoothing spline** | Medium | C² | Medium | Noisy data | Need to choose smoothing |
| **2D spline** | Slow | C² | High | Regular grid data | Requires rectangular grid |

### When to Use Each Method

1. **Linear**: Quick visualization, computing, discrete data sets
2. **Cubic Spline**: Default choice for smooth data
3. **Barycentric Polynomial**: Need high accuracy, well-behaved function, small dataset
4. **Smoothing Spline**: Noisy experimental data
5. **2D Spline**: Regular grid data in 2D (images, meshes, etc.)

### Common Pitfalls and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Extrapolation explodes | Polynomial instability | Don't extrapolate! Use only within data range |
| Interpolation has wiggles | Oscillations from high-order method | Use linear or smoothing spline |
| Derivatives are noisy | Computing derivatives of noisy data | Use smoothing spline with appropriate s |
| Overfitting | Too much emphasis on exact fit | Use smoothing spline with moderate s |
| Discontinuous derivative | Using linear interpolation | Switch to cubic spline |

---

