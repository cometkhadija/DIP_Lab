import numpy as np
from scipy.interpolate import interp1d

# Original points from PDF example
x = np.array([1, 2, 3, 4])
f_x = np.array([2, 3, 1.5, 2.5])

# Target: enlarge to 8 points
new_len = 8
x_new = np.linspace(x.min(), x.max(), new_len)

# Nearest neighbor interpolation
interp_nearest = interp1d(x, f_x, kind='nearest')
f_nearest = interp_nearest(x_new)

# Linear interpolation
interp_linear = interp1d(x, f_x, kind='linear')
f_linear = interp_linear(x_new)

# Cubic interpolation
interp_cubic = interp1d(x, f_x, kind='cubic')
f_cubic = interp_cubic(x_new)

# Manual linear example from PDF (lambda=2/7 between x2 and x3)
lambda_val = 2/7
F_manual = (1 - lambda_val) * f_x[1] + lambda_val * f_x[2]

# Outputs
print("New x:", x_new)
print("Nearest:", f_nearest)
print("Linear:", f_linear)
print("Cubic:", f_cubic)
print(f"Manual linear at lambda=2/7: {F_manual}")

# Another example for lambda=4/7
lambda_val2 = 4/7
F_manual2 = (1 - lambda_val2) * f_x[2] + lambda_val2 * f_x[3]
print(f"Manual linear at lambda=4/7: {F_manual2}")
