from orientation import * 
import numpy as np


Q = [0.707, 0.707, 0.0, 0.0]  # Quaternion representing a 90-degree rotation around x-axis
theta, k = quat2Theta_k(Q)

print("\nRotation Angle (theta) in rad:", theta)  # hould be approximately 1.571 (90 degrees in radians)
print("\nRotation Axis (k):", k)          # Should be [1.0, 0.0, 0.0] cause the rotation is around x-axis

# state_elogdot 3-dimensional NumPy array
state_elogdot = np.array([0.1, 0.2, 0.3])  # Example state of the logarithmic map

omega = logDerivative2_AngleOmega(state_elogdot, Q)
print("\nComputed Angular Velocity (omega):", omega)


