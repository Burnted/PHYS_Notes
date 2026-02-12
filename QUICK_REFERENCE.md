# Physics 234: Quick Reference Guide

## Essential Functions at a Glance

### Math & Trigonometry
```python
import math
import numpy as np

# Trigonometry (use radians!)
sin_val = math.sin(angle_rad)
cos_val = math.cos(angle_rad)
tan_val = np.tan(angle_rad)

# Conversions
degrees_to_radians = deg * math.pi / 180
radians_to_degrees = rad * 180 / math.pi

# Common values
math.pi        # π
math.e         # Euler's number
math.sqrt(x)   # Square root
math.factorial(n)  # n!
```

### Array Operations (NumPy)
```python
import numpy as np

# Create arrays
x = np.linspace(0, 10, 100)          # 100 points from 0 to 10
x = np.logspace(0, 3, 50)             # Log-spaced: 10^0 to 10^3
x = np.array([1, 2, 3, 4, 5])        # Explicit array
x = np.arange(0, 10, 0.1)             # Start, stop, step

# Element-wise operations
y = np.sin(x)
y = np.sqrt(x)
y = x**2 + 3*x + 1

# Array statistics
mean = np.mean(x)
std = np.std(x)
max_val = np.max(x)
sum_val = np.sum(x)
```

### Integration & Derivatives
```python
from scipy.integrate import quad, dblquad
from scipy.integrate import odeint  # For ODEs

# 1D Integration (adaptive, automatic error control)
def f(x):
    return np.sin(x)

result, error = quad(f, 0, np.pi)

# 2D Integration
def f_2d(y, x):
    return x**2 + y**2

result, error = dblquad(f_2d, 0, 1, lambda x: 0, lambda x: 1)

# Numerical Derivative (from Notebook 03)
dx = 5e-6
df_dx = (f(x + dx) - f(x - dx)) / (2 * dx)
```

### Optimization & Fitting
```python
from scipy.optimize import fsolve, minimize, curve_fit

# Find roots of f(x) = 0
root = fsolve(f, x0=1.0)

# Minimize function
result = minimize(f, x0=1.0)
x_min = result.x
f_min = result.fun

# Fit model to data
def model(x, a, b, c):
    return a * x**2 + b*x + c

params, covariance = curve_fit(model, x_data, y_data)
uncertainties = np.sqrt(np.diag(covariance))
```

### Interpolation
```python
from scipy.interpolate import interp1d, UnivariateSpline

# Linear interpolation
f_linear = interp1d(x_data, y_data, kind='linear')
y_interp = f_linear(x_new)

# Cubic spline
f_cubic = interp1d(x_data, y_data, kind='cubic')

# Smoothing spline (for noisy data)
spline = UnivariateSpline(x_data, y_data, s=2.0)
y_smooth = spline(x_data)

# Evaluate derivatives of spline
derivative = spline.derivative()(x_new)
integral = spline.integral(x_min, x_max)
```

### Physical Constants
```python
from scipy import constants

# Common constants
k_coulomb = 1 / (4 * np.pi * constants.epsilon_0)
epsilon_0 = constants.epsilon_0
mu_0 = constants.mu_0
c = constants.c  # speed of light
G = constants.G  # gravitational constant
k_B = constants.k  # Boltzmann constant

# Useful conversions
charge_C = 50e-6  # 50 μC in Coulombs
```

### Plotting
```python
import matplotlib.pyplot as plt

# Basic plots
plt.plot(x, y, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Log-log plot (for convergence analysis)
plt.loglog(N, error)
plt.xlabel('Number of points')
plt.ylabel('Error')

# Multiple subplots
fig, axes = plt.subplots(1, 2)
axes[0].plot(x1, y1)
axes[1].plot(x2, y2)
```

---

## Notebook Progression & Key Topics

### Notebook 01: Calculator
- **Goal**: Solve physics problems numerically
- **Example**: Projectile motion
- **Key Skills**: Variables, libraries, basic calculations

### Notebook 02: Representations
- **Goal**: Understand how computers store numbers
- **Key Concepts**: Binary, hexadecimal, two's complement
- **Functions**: bin(), hex(), oct()

### Notebook 03: Derivatives & Integrals
- **Goal**: Master numerical calculus
- **Derivative Formula**: $\frac{df}{dx} \approx \frac{f(x+dx) - f(x-dx)}{2dx}$ with dx ≈ 5×10^-6
- **Integration**: Riemann sums, then quad()
- **Key Insight**: Error ~ 1/N for basic methods, better methods have steeper convergence

### Notebook 04: Force on a Charge
- **Goal**: Apply numerical integration to physics
- **Method**: Calculate potential, then take gradient
- **Tools**: dblquad() for 2D integration, numerical derivatives
- **Physics**: Coulomb's law, superposition, electric potential

### Notebook 05: Magnetic Forces
- **Goal**: Handle problems without analytical solutions
- **Approach**: Experiment → fit → predict
- **Lesson**: When to compute vs. when to measure

### Notebook 06: Solving & Fitting
- **Goal**: Find solutions and fit models
- **Tools**: fsolve(), minimize(), curve_fit()
- **Application**: Parameter extraction from data

### Notebook 07: Interpolation
- **Goal**: Estimate function values from data
- **Methods**: Linear, polynomial, cubic spline
- **Warning**: Extrapolation is dangerous!

---

## Common Mistakes & Best Practices

### Mistakes to Avoid
1. **Degrees vs. Radians**: All trig functions use radians
2. **Step Size Too Small**: In derivatives, very small dx causes roundoff error
3. **Extrapolating Polynomials**: They explode outside data range
4. **Ignoring Units**: Always track units in calculations
5. **Not Validating**: Compare to analytical solutions or known limits

### Best Practices
1. **Start Simple**: Test with analytical solutions first
2. **Convergence Study**: Check how results change with resolution
3. **Use Libraries**: Don't write integration/optimization code from scratch
4. **Document Units**: Write them in variable names or comments
5. **Check Dimensions**: Verify results make physical sense
6. **Error Estimates**: Always report uncertainties with results

---

## Problem-Solving Flowchart

1. **Understand the Physics**
   - What are the governing equations?
   - Are analytical solutions available?
   - What approximations apply?

2. **Design the Computation**
   - Which numerical method is appropriate?
   - What is the expected accuracy?
   - How do I validate the result?

3. **Implement**
   - Write clean, documented code
   - Use library functions where available
   - Test with known cases first

4. **Verify**
   - Check analytical limits
   - Perform convergence studies
   - Look for physical unreasonableness

5. **Optimize**
   - Improve accuracy if needed
   - Optimize speed if needed
   - Document results clearly

---

## Physical Constants Lookup

| Constant | Symbol | Value |
|----------|--------|-------|
| Speed of light | c | 2.998×10^8 m/s |
| Gravitational constant | G | 6.674×10^-11 m³/(kg·s²) |
| Coulomb constant | k | 8.99×10^9 N·m²/C² |
| Permittivity of free space | ε₀ | 8.854×10^-12 F/m |
| Permeability of free space | μ₀ | 1.257×10^-6 H/m |
| Planck's constant | h | 6.626×10^-34 J·s |
| Boltzmann constant | k_B | 1.381×10^-23 J/K |
| Elementary charge | e | 1.602×10^-19 C |
| Electron mass | m_e | 9.109×10^-31 kg |
| Proton mass | m_p | 1.673×10^-27 kg |

---

## Important Equations Reference

**Projectile Motion**
$$x(t) = x_0 + v_0 t + \frac{1}{2}gt^2$$

**Coulomb's Law**
$$\vec{F} = k\frac{q_1 q_2}{r^2}\hat{r}$$

**Electric Potential**
$$V(r) = \frac{1}{4\pi\epsilon_0}\frac{q}{r}$$

**Numerical Derivative (Symmetric)**
$$\frac{df}{dx} \approx \frac{f(x+dx) - f(x-dx)}{2dx}$$

**Numerical Integration (Riemann)**
$$\int_a^b f(x)dx \approx \sum_{i=0}^{N} f(x_i)\frac{b-a}{N}$$

---


