# Physics: Computational Physics - Notebook Summaries

## Overview
This document provides comprehensive summaries of the Physics notebooks, detailing key concepts, important functions, and practical usage examples.

---

## 01 Calculator.ipynb

### Topics Covered
**Primary Focus: Translating Physics Problems into Numerical Solutions**

This notebook introduces the foundational philosophy of computational physics—moving from analytical solutions to numerical computation. It demonstrates how to:
- Use a computer as a calculator to solve real physics problems
- Extract numerical results using both analytical formulas and numerical methods
- Bridge the gap between mathematical solutions and practical computations

### Key Concepts
1. **Analytical vs. Numerical Solutions**: The course emphasizes that while analytical solutions are valuable, they're no longer the only way to extract quantitative results from physics problems
2. **Constant Acceleration Kinematics**: A projectile motion problem serves as the foundational physics example
3. **Python Basics**: Introduction to variables, libraries, and output in Python
4. **Plotting and Visualization**: Creating graphs of physical quantities as functions of time

### Important Physics Problem: Projectile Motion
**Problem Statement**: A ball is thrown at 140 km/h, 30° east of north, at 40° elevation from a height of 2 meters. Find when and where it hits the ground.

**Key Physics Equations**:
- Force on object: $\vec{F} = m\vec{g}$
- Newton's second law: $\vec{F} = m\vec{a}$, therefore $\vec{a} = \vec{g}$
- Velocity integration: $\vec{v} = \vec{v_0} + \vec{g}t$
- Position integration: $\vec{x} = \vec{x_0} + \vec{v_0}t + \frac{1}{2}\vec{g}t^2$

The z-component gives a quadratic equation for impact time:
$$0 = x_{0,z} + v_{0,z}t + \frac{1}{2}gt^2$$

### Important Functions & Examples

#### 1. **math.sin() and math.cos()**
```python
import math

# Convert angle from degrees to radians for trig functions
angle_degrees = 40.0
angle_radians = angle_degrees / 180 * math.pi
sin_value = math.sin(angle_radians)

# Example: Calculate vertical component of initial velocity
v_0z = 140 * 1000./3600 * math.sin(40.0/180*math.pi)  # Convert km/h to m/s
```
**How to Use**: Trigonometric functions require radians, not degrees. Always convert degrees to radians using `deg * π / 180`.
**Key Points**: 
- math.pi is available in the math library
- Common angles: 30°=π/6, 45°=π/4, 60°=π/3, 90°=π/2

#### 2. **Basic Variable Assignment**
```python
x_0z = 2          # meters
v_0z = 24.99...   # m/s (result from above calculation)
g = -9.8          # m/s^2 (negative because downward)
```
**How to Use**: Store physical quantities with meaningful names and appropriate units.

---

## 02 Representations.ipynb

### Topics Covered
**Primary Focus: How Computers Represent Numbers and Data**

This notebook bridges physics and computer science, explaining:
- Binary, octal, and hexadecimal number systems
- How 64-bit computers store integers and logical values
- Limitations of computer arithmetic
- Special number representations (Gray code)

### Key Concepts

1. **Binary Numbers**
   - Modern computers use 64-bit (8 bytes) storage
   - Can represent 2^64 ≈ 18.4 billion billion unique states
   - Binary to decimal: sum of powers of 2 where bit is "on"

2. **Integer Representations**
   - **Unsigned integers**: 0 to 2^64-1 for 64 bits
   - **Signed integers**: Two's complement notation allows negative numbers
   - Python handles arbitrary-precision integers (unlimited size)

3. **Logical Variables**
   - True/False stored as 1-bit values
   - Can be represented as 0x1 (true) or 0x0 (false) in hexadecimal

4. **Gray Code**
   - Alternate binary representation where only one bit changes between consecutive numbers
   - Useful for rotary encoders and analog-to-digital converters
   - Formula: `gray = n ^ (n >> 1)` where ^ is XOR and >> is right-shift

### Important Functions & Examples

#### 1. **bin(), oct(), hex() - Number Base Conversions**
```python
# Convert to different bases
number = 99
print(bin(number))  # Output: 0b1100011 (binary)
print(oct(number))  # Output: 0o143 (octal, base 8)
print(hex(number))  # Output: 0x63 (hexadecimal, base 16)

# Also works with very large numbers
large = 2**64 - 1
print(hex(large))   # Output: 0xffffffffffffffff (all bits on)
```
**How to Use**: Essential for understanding computer storage and debugging
**Key Points**:
- `0b` prefix = binary, `0o` = octal, `0x` = hexadecimal
- Hexadecimal is compact: each hex digit = 4 binary digits

#### 2. **math.factorial() - Arbitrary Precision Arithmetic**
```python
import math

# Python can handle very large factorials
factorial_65 = math.factorial(65)  # Result: 8247650592082...
factorial_200 = math.factorial(200)  # Incredibly large number

# Can be converted to string to see digit count
str_form = str(math.factorial(200))
digit_count = len(str_form)  # 375 digits!
```
**How to Use**: Demonstrates Python's unlimited integer precision
**Key Points**: Unlike many languages, Python doesn't overflow with large integers

#### 3. **Bitwise Operations - Gray Code Example**
```python
# Gray code conversion
n = 5
gray = n ^ (n >> 1)  # ^ is XOR, >> is right shift
# ^ = bitwise exclusive OR (1 if bits differ, 0 if same)
# >> = right shift (divide by 2, rounded down)

# Example for range of numbers
for i in range(32):
    gray = i ^ (i // 2)
    print(f"{i}: {bin(gray)}")
```
**How to Use**: Bitwise operations for efficient number manipulation
**Key Points**:
- XOR (^): both 0 or both 1 → 0; one 0, one 1 → 1
- Right-shift (>>): equivalent to integer division by 2

---

## 03 Derivatives and Integrals.ipynb

### Topics Covered
**Primary Focus: Numerical Methods for Calculus**

This notebook implements numerical techniques for computing derivatives and integrals—core skills for computational physics.

### Key Concepts

1. **Numerical Derivatives**
   - **Best Practice Rule**: Use symmetric difference formula with optimal step size
   - Formula: $\frac{df}{dx} \approx \frac{f(x+dx) - f(x-dx)}{2dx}$
   - Optimal step size: dx ≈ 5×10^-6 × |f/f'| for typical functions
   - Two-step method (calculate xp and xm separately) reduces roundoff error

2. **Numerical Integration (Riemann Sums)**
   - Definition: $\int_a^b f(x)dx = \lim_{N \to \infty} \sum_{i=0}^N f(x_i) \cdot \frac{b-a}{N}$
   - Basic implementation: rectangular rule (width × height for each rectangle)
   - Convergence: Error decreases roughly as 1/N

3. **Better Integration Methods**
   - Trapezoidal rule: More accurate than rectangles
   - Simpson's rule: Uses parabolic approximation
   - Gaussian quadrature: Optimal point placement for accuracy
   - Romberg integration: Recursive refinement

### Important Functions & Examples

#### 1. **numpy.linspace() - Creating Arrays**
```python
import numpy as np

# Create linearly spaced array
x = np.linspace(0, np.pi/2, 100, endpoint=False)
# 0 to π/2 in 100 points, excluding endpoint

# Useful for integration
# endpoint=False: total of N evenly spaced values
x = np.linspace(0, np.pi/2, 10, endpoint=False)
# Points: [0, π/20, 2π/20, ..., 9π/20]
```
**How to Use**: Standard way to create arrays for numerical integration and derivatives
**Key Points**:
- `endpoint=False` gives exactly N points
- Default `endpoint=True` includes both endpoints (N points total)

#### 2. **numpy.sum() - Array Summation**
```python
import numpy as np

# Riemann sum for integration
x = np.linspace(0, np.pi/2, N, endpoint=False)
y = np.sin(x)  # Function values
dx = np.pi/2 / N  # Step size

integral_approx = np.sum(y) * dx  # Sum of (height × width)
```
**How to Use**: Efficient summation over arrays
**Key Points**: numpy operations are much faster than Python loops

#### 3. **scipy.integrate.quad() - Adaptive Integration**
```python
from scipy.integrate import quad
import numpy as np

def f(x):
    return np.sin(x)

# quad automatically adapts to desired accuracy
result, error = quad(f, 0, np.pi/2)
# result ≈ 1.0 (analytical answer)
# error ≈ 1.1e-14 (estimated error)
```
**How to Use**: Production-quality integration with automatic error estimation
**Key Points**:
- Returns tuple: (result, error_estimate)
- Much more efficient than manual Riemann sums
- Can handle singularities and oscillations

#### 4. **Plotting Integration Convergence**
```python
import numpy as np
import matplotlib.pyplot as plt

# Log-log plot shows convergence rate
N = np.logspace(1, 8, 8)  # 10, 100, 1000, ..., 100M points
errors = []

for n in N:
    result = integral(f, int(n))
    error = abs(1 - result)
    errors.append(error)

plt.loglog(N, errors)
plt.xlabel('Number of points')
plt.ylabel('Error |1 - Integral|')
```
**How to Use**: Understand how numerical methods improve with resolution
**Key Points**: 
- Slope on log-log plot shows convergence rate
- Riemann sum: slope ≈ -1 (error ~ 1/N)
- Better methods have steeper slopes

---

## 04 Force on a Charge.ipynb

### Topics Covered
**Primary Focus: Numerical Integration for Physics Applications**

This notebook applies numerical integration to a real electrostatics problem: finding the force on a test charge due to a uniformly charged cylinder.

### Key Physics Concepts

1. **Coulomb's Law**: The fundamental equation for electric force
   - Force between point charges: $\vec{F} = k \frac{q_1 q_2}{r^2} \hat{r}$
   - k = 1/(4πε₀) ≈ 8.99 × 10⁹ N⋅m²/C²

2. **Electric Field and Superposition**
   - $\vec{E} = \vec{F}/q$ (field per unit charge)
   - Total field from extended charge distribution: integral over surface
   - $\vec{E}(r) = \frac{1}{4\pi\epsilon_0} \int \frac{dq(\vec{r}-\vec{r'})}{|\vec{r}-\vec{r'}|^3}$

3. **Electric Potential** (used in this problem)
   - Scalar field: $V(r) = \frac{1}{4\pi\epsilon_0} \int \frac{dq}{|\vec{r}-\vec{r'}|}$
   - Easier to integrate (scalar vs vector)
   - Field recovered by gradient: $\vec{E} = -\nabla V$

### Problem Setup

**Geometry:**
- Insulating cylinder: L = 1 m, diameter = 2 cm (radius = 0.01 m)
- Cylinder axis along z-axis, centered at origin
- Surface charge density σ = 100 μC uniformly distributed
- Test charge q = 50 μC at arbitrary position

### Numerical Strategy

1. **Parameterize cylinder surface** using angle θ and z-position
2. **Calculate potential** at test charge from each surface element
3. **Integrate** using 2D integration (dblquad) over surface
4. **Take gradient numerically** to find electric field components
5. **Multiply by test charge** to find force

### Important Functions & Examples

#### 1. **scipy.integrate.dblquad() - 2D Integration**
```python
from scipy.integrate import dblquad

# Integrate f(y,x) over region
# x ranges from a to b
# for each x, y ranges from g(x) to h(x)
result, error = dblquad(f, a, b, lambda x: g(x), lambda x: h(x))

# Example: Integrate over cylinder surface
def potential_element(theta, z):
    """Potential contribution from surface element at angle theta, height z"""
    # position on cylinder surface in cylindrical coords
    r_source = R * np.exp(1j * theta)  # position of charge element
    z_source = z
    
    # distance to test charge
    distance = ...
    
    # contribution to potential (charge * 1/distance)
    return sigma * R * dz * dtheta / distance

# Integrate over full cylinder surface
# z from -L/2 to L/2
# theta from 0 to 2*pi
potential, error = dblquad(potential_element, 0, 2*np.pi, 
                          lambda th: -L/2, lambda th: L/2)
```
**How to Use**: Double integration over 2D surfaces
**Key Points**:
- Inner integral is over second parameter (y if function is f(y,x))
- Outer integral is over first parameter (x)
- Limits of inner integral can depend on outer variable
- Very efficient for smooth functions

#### 2. **Numerical Derivative of Potential**
```python
# Remember from Notebook 03: use symmetric difference formula
def electric_field_component(V_func, position, component='x', dx=1e-5):
    """Calculate one component of E = -dV/dx using symmetric difference"""
    
    pos_plus = position.copy()
    pos_minus = position.copy()
    
    if component == 'x':
        idx = 0
    elif component == 'y':
        idx = 1
    elif component == 'z':
        idx = 2
    
    pos_plus[idx] += dx
    pos_minus[idx] -= dx
    
    V_plus = V_func(pos_plus)
    V_minus = V_func(pos_minus)
    
    E = -(V_plus - V_minus) / (2 * dx)
    return E
```
**How to Use**: Convert potential field to electric field
**Key Points**: Same symmetric difference rule from Notebook 03 applies

#### 3. **scipy.constants - Physical Constants**
```python
from scipy import constants

# Coulomb constant
k = 1 / (4 * np.pi * constants.epsilon_0)

# Charge conversions
charge_microC = 50e-6  # 50 μC in Coulombs

# Can also use direct values
epsilon_0 = constants.epsilon_0  # ~8.854e-12 F/m
c = constants.c  # speed of light: 299,792,458 m/s
```
**How to Use**: Access physical constants consistently
**Key Points**: Avoids hardcoding values, ensures consistency with NIST standards

### Validation Strategy

1. **Far-field limit**: At large distances, should match point charge behavior
   - All charge appears concentrated at origin
   - Force should follow Coulomb's law: $F = k q Q / r^2$

2. **Symmetry checks**:
   - Force on equatorial plane (z=0) should be radially symmetric
   - No z-component force should exist by symmetry

3. **Analytical solution on z-axis**:
   - For test charge on cylinder axis, can solve analytically
   - Compare numerical integration to analytical result

---

## 05 Numerical Calculation of Forces Between Magnets.ipynb

### Topics Covered
**Primary Focus: When Analytical Solutions Don't Exist**

This notebook demonstrates how to handle real physical problems that don't have closed-form analytical solutions.

### Key Concepts

1. **Magnetic Dipole Moment**: How to characterize a magnet
   - Dipole moment $\vec{m}$ (in A⋅m²)
   - Can be measured experimentally

2. **Magnetic Field from Dipole**:
   - Along axis: $B_z \approx \frac{\mu_0}{4\pi} \frac{2m}{z^3}$ (near field)
   - General formula involves complex vector calculations

3. **Forces on Magnetic Objects**:
   - Force ∝ gradient of field: $\vec{F} \propto \nabla(\vec{m} \cdot \vec{B})$
   - Real magnets can't be approximated as simple dipoles at very close range
   - Experimental measurement becomes necessary

### Numerical Approach

Since analytical solutions for real magnets are intractable:
1. **Measure force experimentally** using precise balance
2. **Fit model parameters** to experimental data
3. **Use fitted model** for force predictions in regime of interest

### Important Concept: When to Use Computation

This notebook illustrates the pragmatic approach to physics:
- **If analytical solution exists**: derive and use it
- **If numerical integration works**: implement it (like Notebook 04)
- **If neither feasible**: measure experimentally, then fit models

---

## 06 Solving, Minimizing, Fitting.ipynb

### Topics Covered
**Primary Focus: Solving Equations and Fitting Data to Models**

Advanced numerical techniques for optimization and parameter estimation.

### Key Concepts

1. **Root Finding**: Solving f(x) = 0
   - **Newton-Raphson Method**: Uses derivative for fast convergence
   - **Bisection Method**: Slow but reliable, needs bracketing
   - **Secant Method**: Similar to Newton-Raphson but no derivative needed

2. **Function Minimization**
   - Finding minimum of f(x)
   - Applications: energy minimization, cost function optimization
   - Can solve equations using: find minimum of |f(x)|
