# Comprehensive Guide to Interpolation

## Table of Contents
1. [Introduction & Motivation](#introduction--motivation)
2. [Key Concepts](#key-concepts)
3. [Linear Interpolation](#linear-interpolation)
4. [Spline Interpolation](#spline-interpolation)
5. [Polynomial/Barycentric Interpolation](#polynomialbarycentric-interpolation)
6. [Chebyshev Interpolation](#chebyshev-interpolation)
7. [2D Interpolation](#2d-interpolation)
8. [Practical Tips & Best Practices](#practical-tips--best-practices)

---

## Introduction & Motivation

Interpolation is the process of estimating values between known data points. You might need interpolation in two common scenarios:

1. **Data Interpolation**: You have discrete data points and need to evaluate values between them
2. **Function Approximation**: Evaluating a function is computationally expensive (e.g., numerical integration), so you compute it at discrete points, then use interpolation to approximate it elsewhere

### Example Scenario
If calculating an electric or magnetic field takes several minutes per point, you can:
- Calculate the field at a grid of points
- Use interpolation to quickly estimate the field at other locations
- Save the interpolated function to file for later use

---

## Key Concepts

### What Makes a Good Interpolation?

| Property | Importance | How to Achieve |
|----------|-----------|-----------------|
| **Accuracy** | Must closely match actual function | More data points, better interpolation method |
| **Smoothness** | Continuous derivatives | Splines or polynomial methods |
| **Efficiency** | Fast evaluation time | Pre-computed coefficients |
| **Stability** | No wild oscillations | Chebyshev nodes instead of uniform spacing |

### The Trade-offs

```
Linear Interpolation
├─ Pro: Simple, stable, continuous
└─ Con: Not smooth (discontinuous derivative)

Spline Interpolation
├─ Pro: Smooth (continuous 2nd derivative), stable
└─ Con: Slightly more complex, depends on non-local values

Polynomial Interpolation
├─ Pro: Can achieve very high accuracy
└─ Con: Can oscillate wildly at boundaries (Runge's phenomenon)

Chebyshev Interpolation
├─ Pro: Near-optimal accuracy, minimizes oscillations
└─ Con: Requires Chebyshev nodes (non-uniform spacing)
```

---

## Linear Interpolation

### Mathematical Basis

For two neighboring points $(x_i, y_i)$ and $(x_{i+1}, y_{i+1})$, linear interpolation finds a point on the line connecting them:

$$y(x) = \frac{y_{i+1} - y_i}{x_{i+1} - x_i}(x - x_i) + y_i$$

### How to Use (NumPy)

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a function and create a table of values
def sinc(x):
    return np.where(x==0, np.ones_like(x), np.sin(x)/x)

# Create coarse table of values
table_x = np.linspace(-4*np.pi, 4*np.pi, 25)  # 25 points
table_y = sinc(table_x)

# Create fine grid where we want to interpolate
x = np.linspace(-4*np.pi, 4*np.pi, 201)

# Linear interpolation
y_interp = np.interp(x, table_x, table_y)

# Visualize
plt.plot(table_x, table_y, 'o', label='Data points')
plt.plot(x, y_interp, label='Linear interpolation')
plt.plot(x, sinc(x), label='True function')
plt.legend()
plt.show()
```

### Key Properties

- **Continuous** but **not smooth** (has discontinuous first derivative)
- Works well for rough approximations
- The derivative of linear interpolation shows visible kinks at data points

### When to Use

- Quick approximations where smoothness doesn't matter
- Small datasets where more complex methods aren't justified
- When computational speed is critical

---

## Spline Interpolation

### Mathematical Concept

A spline is a piecewise polynomial that ensures:
- **Continuity**: Function matches at data points
- **Smooth derivatives**: First and second derivatives are continuous

This is achieved by:
1. Using a cubic polynomial between each pair of data points
2. Imposing the condition that the **second derivative is continuous**
3. Solving for the unknown second derivatives at each data point

### Important Property

> The second derivative is *linearly* interpolated between data points. This seems odd but produces a mathematically elegant solution requiring cubic polynomials between points.

### How to Use (SciPy)

```python
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt

# Create table of values
table_x = np.linspace(-4*np.pi, 4*np.pi, 25)
table_y = sinc(table_x)

# Create spline object
spline = make_interp_spline(table_x, table_y, k=3)  # k=3 means cubic

# Evaluate at many points
x = np.linspace(-4*np.pi, 4*np.pi, 201)
y_interp = spline(x)

# Spline is callable like a function
plt.plot(table_x, table_y, 'o', label='Data points')
plt.plot(x, y_interp, label='Cubic spline')
plt.plot(x, sinc(x), label='True function')
plt.legend()
plt.show()
```

### Key Parameters

- **`k`**: Degree of the spline
  - `k=1`: Linear spline (piecewise linear)
  - `k=3`: Cubic spline (most common, smooth 2nd derivative)
  - `k=5`: Quintic spline (smoother 3rd and 4th derivatives)

### Why Splines Are Great

- Generally more accurate than linear interpolation
- Second derivative is continuous (smooth enough for most physics)
- Third derivative is discontinuous but usually unnoticed
- Values depend on ALL data points (global dependence), which can be desirable
- Stable: no wild oscillations like polynomial interpolation

### Limitations

- Second derivative itself is discontinuous
- Non-local: changing one data point affects the entire interpolant
- More computationally expensive than linear interpolation

---

## Polynomial/Barycentric Interpolation

### Mathematical Concept

There exists a unique polynomial of degree $N-1$ that passes through $N$ data points. The **Lagrange formula** finds it:

$$p(x) = \sum_{i=0}^{N} f(x_i) \prod_{k=0, k \neq i}^{N} \frac{x - x_k}{x_i - x_k}$$

However, this is numerically unstable. The **Barycentric formula** is more stable:

$$p(x) = \frac{\sum_{i=0}^{N} f_i \frac{w_i}{x - x_i}}{\sum_{i=0}^{N} \frac{w_i}{x - x_i}}$$

where the weights $w_i$ depend only on the $x$-values (data points), not the function values.

### How to Use (SciPy)

```python
from scipy.interpolate import BarycentricInterpolator
import numpy as np
import matplotlib.pyplot as plt

# Create table of values
table_x = np.linspace(-4*np.pi, 4*np.pi, 25)
table_y = sinc(table_x)

# Create barycentric interpolator
bary = BarycentricInterpolator(table_x, table_y)

# Evaluate at many points
x = np.linspace(-4*np.pi, 4*np.pi, 201)
y_interp = bary(x)

plt.plot(table_x, table_y, 'o', label='Data points')
plt.plot(x, y_interp, label='Polynomial interpolation')
plt.plot(x, sinc(x), label='True function')
plt.legend()
plt.show()
```

### The Problem: Runge's Phenomenon

When using **uniformly spaced points**, polynomial interpolation can oscillate wildly, especially near the boundaries:

```python
# Bad: Uniform spacing leads to oscillations
uniform_x = np.linspace(-4*np.pi, 4*np.pi, 25)
bary_bad = BarycentricInterpolator(uniform_x, sinc(uniform_x))

x = np.linspace(-4*np.pi, 4*np.pi, 201)
plt.plot(x, bary_bad(x) - sinc(x))  # Big errors at ends!
plt.title("Error with uniform spacing")
plt.show()
```

### The Solution: Chebyshev Nodes

Instead of uniform spacing, use **Chebyshev nodes** where points are denser at the boundaries:

$$x_i = a \cos\left(\frac{i \pi}{N}\right), \quad i = 0, 1, \ldots, N$$

This dramatically reduces oscillations and is nearly optimal!

```python
# Good: Chebyshev spacing minimizes oscillations
N = 25
cheby_x = 4*np.pi * np.cos(np.linspace(0, N, N+1) * np.pi / N)
bary_good = BarycentricInterpolator(cheby_x, sinc(cheby_x))

x = np.linspace(-4*np.pi, 4*np.pi, 201)
plt.plot(x, bary_good(x) - sinc(x))  # Much smaller errors!
plt.title("Error with Chebyshev spacing")
plt.show()
```

### Advantages of Chebyshev Nodes

1. **Near-optimal**: Produces the polynomial closest to the "minimax polynomial" (smallest maximum deviation)
2. **Truncatable**: If you compute high-order and truncate to lower order, you still get nearly optimal polynomial
3. **Minimal oscillation**: Runge's phenomenon is essentially eliminated

### When to Use Polynomial Interpolation

- Need high accuracy with smooth functions
- Can use Chebyshev nodes (non-uniform spacing)
- Function has no discontinuities in derivative
- Willing to trade computational complexity for accuracy

---

## Chebyshev Interpolation

### Key Advantages Over Standard Polynomial Interpolation

While standard polynomial interpolation (with Chebyshev nodes) works well, true **Chebyshev interpolation** using Chebyshev polynomials offers:

- More stable numerically
- Better handling of ill-conditioned problems
- Can easily adjust approximation order
- Direct access to polynomial coefficients

### The Chebyshev Polynomial Basis

Chebyshev polynomials $T_n(x)$ are defined on $[-1, 1]$:

- $T_0(x) = 1$
- $T_1(x) = x$
- $T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$ (recurrence relation)

### Key Concept: Coordinate Transformation

**Chebyshev interpolation requires your data in the range [-1, 1].**

If your data is in a different range $[a, b]$, transform it:

$$t = \frac{2x - (b+a)}{b-a}$$

### Example with Coordinate Transformation

```python
# Your data is in range [0, 50]
r_list = np.linspace(0, 50, 100)  # Original range
y_list = your_function(r_list)

# Transform to [-1, 1]
r_list_cheb = r_list / 50.0 - 1.0  # Map [0, 50] to [-1, 1]

# Define wrapper function that transforms back
def RCheb(r):
    """Input r is in [-1, 1], return function value in original coordinates"""
    return your_function((1 + r) * 50)  # Map [-1, 1] back to [0, 50]

# Create Chebyshev approximation
from numpy.polynomial.chebyshev import chebinterpolate
c = chebinterpolate(RCheb, 40)  # 40th-degree Chebyshev approximation
```

### Understanding the Transformation

```
FORWARD (Data Space to Chebyshev):
r ∈ [0, 50]  →  r_cheb = r/50.0 - 1.0  →  r_cheb ∈ [-1, 1]
                         ↑
                         Divide by 50 (the range width)
                         Subtract 1 (to shift [-0.5, 0.5] to [-1, 1])

INVERSE (Chebyshev Back to Data Space):
r_cheb ∈ [-1, 1]  →  r = (1 + r_cheb) * 50  →  r ∈ [0, 50]
                            ↑
                            Add 1 (shift [-1, 1] to [0, 2])
                            Multiply by 50 (scale to [0, 100] then take first half... wait)
```

Actually, let me clarify the math more precisely:

```
Linear transformation: t = 2(x - a)/(b - a) - 1
Maps [a, b] → [-1, 1]

For [0, 50]:  t = 2(x - 0)/(50 - 0) - 1 = 2x/50 - 1 = x/25 - 1

Hmm, your code uses: r_cheb = r/50.0 - 1.0

This is slightly different. Let's verify:
- When r = 0:   r_cheb = -1 ✓
- When r = 50:  r_cheb = 0 ✗ (should be 1)

Actually, the standard transformation should be:
r_cheb = 2*r/50 - 1  or equivalently  r_cheb = r/25 - 1

But r/50 - 1 maps [0, 50] to [-1, 0], which might be intentional for a half-range.
If your actual domain is [0, 25], then r/25 - 1 is correct.
```

### Using Chebyshev Interpolation Functions

```python
from numpy.polynomial.chebyshev import (
    chebinterpolate,      # Compute coefficients
    chebpts1,            # Get Chebyshev nodes
    chebval,             # Evaluate 1D
    chebval2d,           # Evaluate 2D
    Chebyshev            # Chebyshev polynomial object
)

# Method 1: Using chebinterpolate (simplest)
c = chebinterpolate(RCheb, 40)  # 40th order approximation
y = np.polynomial.chebyshev.chebval(x_cheb, c)  # Evaluate

# Method 2: Using Chebyshev nodes explicitly
N = 40
x_nodes = chebpts1(N)  # Get the N Chebyshev nodes in [-1, 1]
y_nodes = RCheb(x_nodes)  # Evaluate function at nodes
c = compute_coefficients(x_nodes, y_nodes)  # Compute coefficients
y = chebval(x_cheb, c)  # Evaluate at any point
```

### What Do chebpts1 and chebval Do?

#### `chebpts1(n)`: Get Chebyshev Nodes

```python
from numpy.polynomial.chebyshev import chebpts1

# Get first-kind Chebyshev nodes (also called "roots")
nodes = chebpts1(40)  # Returns 40 points in [-1, 1]

# These are the zeros of the 40th Chebyshev polynomial
# Mathematically: x_i = cos(π(2i+1)/(2n)) for i = 0, 1, ..., n-1
# They are denser at the boundaries, sparser in the middle
```

**Why use Chebyshev nodes?**
- Optimal point selection for polynomial interpolation
- Automatically handle clustering for stability
- Minimize Runge phenomenon

#### `chebval(x, c)`: Evaluate Chebyshev Series

```python
from numpy.polynomial.chebyshev import chebval

# c is array of coefficients [c_0, c_1, c_2, ...]
# Evaluates: f(x) = c_0*T_0(x) + c_1*T_1(x) + c_2*T_2(x) + ...
y = chebval(x, c)  # Works for scalar or array x

# Example:
c = [1, 2, 3]  # Represents 1*T_0(x) + 2*T_1(x) + 3*T_2(x)
x = np.array([-0.5, 0, 0.5])
y = chebval(x, c)  # Returns array of 3 values
```

### Complete Example: Chebyshev Interpolation

```python
import numpy as np
from numpy.polynomial.chebyshev import chebinterpolate, chebval
import matplotlib.pyplot as plt

# Example function (expensive to compute)
def expensive_function(r):
    # Imagine this takes 1 minute to compute
    return np.sin(r) * np.exp(-r/10)

# Step 1: Transform data range
r_original = np.linspace(0, 50, 100)
r_cheb = r_original / 50.0 - 1.0  # Map [0, 50] to [-1, 1]

# Step 2: Define wrapper function for Chebyshev (takes input in [-1, 1])
def RCheb(r):
    """Takes r in [-1, 1], returns function value at original scale"""
    return expensive_function((1 + r) * 25)  # Map back to [0, 50]
    # Note: I'm using 25 here, adjust based on your actual mapping

# Step 3: Compute Chebyshev coefficients
c = chebinterpolate(RCheb, 40)  # Degree 40 approximation

# Step 4: Use interpolation
r_test = np.linspace(0, 50, 1000)
r_test_cheb = r_test / 50.0 - 1.0  # Transform test points

y_chebyshev = chebval(r_test_cheb, c)  # Evaluate
y_actual = expensive_function(r_test)

# Verify accuracy
plt.plot(r_test, y_actual, label='Actual')
plt.plot(r_test, y_chebyshev, label='Chebyshev approx')
plt.legend()
plt.show()

print(f"Max error: {np.max(np.abs(y_actual - y_chebyshev))}")
```

### Convergence with Chebyshev Order

```python
# Test different orders
for n in [10, 20, 30, 40]:
    c = chebinterpolate(RCheb, n)
    y_approx = chebval(r_test_cheb, c)
    error = np.max(np.abs(y_actual - y_approx))
    print(f"Order {n}: Max error = {error:.2e}")

# Typical output:
# Order 10: Max error = 1.23e-4
# Order 20: Max error = 1.45e-7
# Order 30: Max error = 1.89e-10
# Order 40: Max error = 3.21e-13
```

---

## 2D Interpolation

### When You Need 2D Interpolation

Common in physics:
- Electric/magnetic field maps
- Potential energy surfaces
- Temperature distributions
- Any field on a 2D grid

### Method 1: RectBivariateSpline (Easiest)

```python
from scipy.interpolate import RectBivariateSpline
import numpy as np

# Suppose you have computed a function on a 2D grid
x = np.linspace(-1, 1, 40)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)  # Your 2D function

# Create 2D spline interpolator
bvs = RectBivariateSpline(y, x, Z)  # Note: y comes before x!

# Evaluate at new points
x_new = 0.37
y_new = -0.52
value = bvs(y_new, x_new)[0][0]  # Returns [[value]]

# Evaluate on a fine grid
x_fine = np.linspace(-1, 1, 200)
y_fine = np.linspace(-1, 1, 200)
Z_fine = np.array([[bvs(yy, xx)[0][0] for xx in x_fine] 
                    for yy in y_fine])

# Visualize
plt.imshow(Z_fine, extent=[-1, 1, -1, 1], origin='lower')
plt.colorbar()
plt.show()
```

**Important Notes:**
- First argument to `RectBivariateSpline` is $y$ coordinates, second is $x$
- `.shape` of Z must be `(len(y), len(x))`
- Returns `[[value]]`, so access with `[0][0]`

### Method 2: RegularGridInterpolator (More Flexible)

```python
from scipy.interpolate import RegularGridInterpolator

# Create interpolator
rgi = RegularGridInterpolator((x, y), Z.T, method='cubic', bounds_error=False)

# Evaluate at points (note: x, y order here)
points = np.array([[0.37, -0.52], [0.1, 0.2]])
values = rgi(points)
```

### Method 3: 2D Chebyshev Interpolation (Advanced)

For highly accurate approximation of expensive 2D functions:

```python
from numpy.polynomial.chebyshev import chebpts1, chebval2d

# Step 1: Define Chebyshev nodes
Nx, Ny = 40, 39
x_nodes = chebpts1(Nx)  # Points in [-1, 1]
y_nodes = chebpts1(Ny)

# Step 2: Evaluate function on Chebyshev nodes
# F[i,j] = f(x_nodes[i], y_nodes[j])
F = np.array([[expensive_2d_function(x, y) 
               for y in y_nodes] 
              for x in x_nodes])

# Step 3: Compute 2D Chebyshev coefficients
# c[i,j] = coefficient of T_i(x) * T_j(y)

# Helper function
def ChebyshevN(order, N):
    """Create Nth-degree Chebyshev polynomial with 1 at position 'order'"""
    coefs = np.zeros(N)
    coefs[order] = 1
    return np.polynomial.chebyshev.Chebyshev(coefs)

# Evaluate Chebyshev polynomials at nodes
yt = np.array([ChebyshevN(order, Ny)(y_nodes) 
               for order in range(Ny)]).T  # (Ny, Ny)
xt = np.array([ChebyshevN(order, Nx)(x_nodes) 
               for order in range(Nx)])    # (Nx, Nx)

# Compute coefficients (with proper normalization)
c = xt @ F @ yt / Nx / Ny * 4
c[0, :] *= 0.5   # Adjust T_0 terms
c[:, 0] *= 0.5

# Step 4: Evaluate at arbitrary points
x_test = np.linspace(-1, 1, 200)
y_test = np.linspace(-1, 1, 200)
X_test, Y_test = np.meshgrid(x_test, y_test)

Z_approx = chebval2d(X_test, Y_test, c)

plt.imshow(Z_approx)
plt.colorbar()
plt.show()
```

#### 2D Chebyshev Mathematics

The 2D Chebyshev approximation has the form:
$$f(x,y) \approx \sum_{i,j} c_{ij} T_i(x) T_j(y)$$

The coefficients are computed using orthogonality:
$$c_{ij} = \frac{4}{NM} \sum_{k,l} f(x_k, y_l) T_i(x_k) T_j(y_l) \times w_{ij}$$

where weights $w_{ij}$ account for the special cases at $i=0$ or $j=0$.

---

## Practical Tips & Best Practices

### 1. When to Use Each Method

| Scenario | Best Choice | Why |
|----------|------------|-----|
| Rough approximation, speed matters | Linear | Simplest, fastest |
| Smooth function, needs derivatives | Spline | Smooth 2nd derivative, stable |
| Very high accuracy, smooth function | Chebyshev polynomial | Near-optimal, minimal oscillation |
| Mixed smooth/discontinuity | Piecewise + Chebyshev | Break at discontinuity, use Chebyshev on each piece |
| Expensive function evaluation | Chebyshev coefficients saved | Evaluate coefficients once, save, load and evaluate many times |
| 2D grid, moderate accuracy | RectBivariateSpline | Simple, works well |
| 2D function, very high accuracy | 2D Chebyshev | Slower but very accurate |

### 2. Handling Special Cases

#### Non-uniform Grids
If your data doesn't lie on a regular grid, use `BarycentricInterpolator`:
```python
from scipy.interpolate import BarycentricInterpolator
interp = BarycentricInterpolator(x_data, y_data)  # Works with any x spacing
```

#### Functions with Discontinuities
Break into multiple regions:
```python
# E.g., electric field with discontinuity at r = r_0

# Region 1: r < r_0
r1 = np.linspace(0, r_0-0.001, 50)
c1 = chebinterpolate(lambda r: f_inside((1+r)*(r_0-0.001)), 30)

# Region 2: r > r_0
r2 = np.linspace(r_0+0.001, r_max, 50)
c2 = chebinterpolate(lambda r: f_outside((1+r)*(r_max-r_0)), 30)

def interpolate(r):
    if r < r_0:
        return chebval((2*r/(r_0-0.001))-1, c1)
    else:
        return chebval((2*r/(r_max-r_0))-1, c2)
```

#### Data Spanning Many Orders of Magnitude
Interpolate the logarithm:
```python
from scipy.interpolate import RectBivariateSpline

# If Z spans 10^-10 to 10^10
Z_log = np.log(np.abs(Z))
bvs = RectBivariateSpline(y, x, Z_log)

# Evaluate and convert back
Z_approx = np.exp(bvs(y_test, x_test))
```

### 3. Saving and Loading Interpolants

Save computation by storing coefficients:

```python
# One-time expensive computation
c = chebinterpolate(expensive_function, 50)
np.save('chebyshev_coefficients.npy', c)

# Later, load and use
c = np.load('chebyshev_coefficients.npy')
y = chebval(x_test, c)
```

### 4. Testing Interpolation Quality

Always verify your interpolation:

```python
# Compute actual values at test points
y_actual = np.array([expensive_function(x) for x in x_test])

# Compare to interpolation
y_approx = chebval(x_test_cheb, c)

# Quantify error
max_error = np.max(np.abs(y_actual - y_approx))
mean_error = np.mean(np.abs(y_actual - y_approx))
rel_error = np.max(np.abs((y_actual - y_approx) / y_actual))

print(f"Max error: {max_error:.2e}")
print(f"Mean error: {mean_error:.2e}")
print(f"Max relative error: {rel_error:.2e}%")
```

### 5. Choosing Interpolation Order

```python
# Test multiple orders
orders = [5, 10, 15, 20, 25, 30]
errors = []

for order in orders:
    c = chebinterpolate(expensive_func, order)
    y_approx = chebval(x_test_cheb, c)
    error = np.max(np.abs(y_actual - y_approx))
    errors.append(error)
    print(f"Order {order}: {error:.2e}")

# Choose where error stops improving significantly
plt.semilogy(orders, errors)
plt.xlabel('Chebyshev Order')
plt.ylabel('Max Error')
plt.show()
```

### 6. Memory Efficiency

For 2D interpolation on large grids:

```python
# Instead of storing full 2D array:
# Original: 1000x1000 grid = 1M points
# Chebyshev: 40x40 = 1600 coefficients (1000x reduction!)

Z_fine = np.zeros((1000, 1000))  # 8 MB
c = np.zeros((40, 40))            # 12.8 KB

# Evaluate as needed
Z_approx = chebval2d(X_test, Y_test, c)  # Fast on-demand evaluation
```

---

## Summary Table: Interpolation Methods

| Method | Code | Speed | Accuracy | Smoothness | Use Case |
|--------|------|-------|----------|-----------|----------|
| **Linear** | `np.interp()` | Very fast | Poor | None | Quick rough approx |
| **Spline** | `make_interp_spline()` | Fast | Good | C² continuous | Smooth curves |
| **Barycentric** | `BarycentricInterpolator()` | Medium | Excellent | C∞ | Non-uniform grids |
| **Chebyshev** | `chebinterpolate()` | Medium | Excellent | C∞ | High accuracy needed |
| **Rect Spline 2D** | `RectBivariateSpline()` | Fast | Good | C² | 2D regular grids |
| **Chebyshev 2D** | Custom (see guide) | Medium | Excellent | C∞ | 2D high accuracy |

---

## Quick Reference: Common Imports

```python
# Linear
import numpy as np
np.interp(x, xp, fp)

# Splines
from scipy.interpolate import make_interp_spline
spline = make_interp_spline(x, y, k=3)
y_new = spline(x_new)

# Polynomial
from scipy.interpolate import BarycentricInterpolator
bary = BarycentricInterpolator(x, y)
y_new = bary(x_new)

# Chebyshev 1D
from numpy.polynomial.chebyshev import chebinterpolate, chebval
c = chebinterpolate(func, n)
y = chebval(x, c)

# Chebyshev nodes
from numpy.polynomial.chebyshev import chebpts1
x_nodes = chebpts1(n)

# 2D Spline
from scipy.interpolate import RectBivariateSpline
bvs = RectBivariateSpline(y, x, Z)
z = bvs(y_new, x_new)[0][0]

# 2D Chebyshev
from numpy.polynomial.chebyshev import chebval2d
Z_approx = chebval2d(X, Y, c)
```


