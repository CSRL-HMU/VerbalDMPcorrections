import roboticstoolbox as rt
import numpy as np

# import rtde_receive
# import rtde_control
import math
import os
import scipy.io
import time
from dmpR3 import *
from dmp import *
import pathlib
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
import threading
import time

# set initial configuration. Initial position of the robotic arm
q0_c = np.array(
    [-1.53928739, -2.47098269, -1.27327633, -1.57354273, 1.6510781, 11.8908723]
)

q0 = q0_c.copy()

# Initialize q with the initial configuration q0. q then will be updated in the control loop with qdot over each iteration.
q = q0.copy()

# declare math pi
pi = math.pi

# define the robot with its DH parameters
ur = rt.DHRobot(
    [
        rt.RevoluteDH(d=0.15185, alpha=pi / 2),
        rt.RevoluteDH(a=-0.24355),
        rt.RevoluteDH(a=-0.2132),
        rt.RevoluteDH(d=0.13105, alpha=pi / 2),
        rt.RevoluteDH(d=0.08535, alpha=-pi / 2),
        rt.RevoluteDH(d=0.0921),
    ],
    name="UR3e",
)

# ur = rt.models.UR5()

# Control cycle
dt = 0.002

# get initial end effector position
g0 = ur.fkine(q0)
R0 = np.array(g0.R)
p0 = np.array(g0.t)

# Initial time
t = 0.0

# start logging
plog = p0
tlog = t
pdlog = p0  # or pdlog = np.array(p0).reshape(1,3)  #initialize pdlog to log the desired position over time

# initialize qdot
qdot = np.zeros(6)

# initialize variables. System's state
# model = dmpR3(40, 10.0)   #setting a dmpR3 model (40 basis functions, 10.0 time constant) -> (N_in, T_in)

# load data total motion
# train dmp model

# Loading the motion (data) and training the DMP model
folderPath = pathlib.Path(__file__).parent.resolve()

dataFile = folderPath / "test_motion_demo.mat"
data = scipy.io.loadmat(str(dataFile))

x_train = data["p"]

wanted_x_train = x_train[1:, :]  # starts from the second row - till the end

p_train = np.array(np.transpose(wanted_x_train))

t_train = np.array(list(range(p_train[1, :].size))) * dt

model = dmpR3(
    20, t_train[-1]
)  # estw oti ayto to einai to model moy anti gia to tyxaio model poy orisame prin -> model = dmpR3(40, 10.0)
model.train(dt, p_train, False)  # training the DMP model


model.set_init_state(p0)  # or p = np.array(rtde_c.getActualTCPPose()[:3]) ??
pd = p0.copy()  # Initially, setting the desired position to the initial position
dot_pd = np.zeros(3)
z = 0.0  # phase variable (used as a vector). Changed it from np.zeros(3) to 0.0 otherwise got error in get_state_dot.

# x in get_state_dot was being passed as an array when it should be a single scalar value.
# This issue could arise if z (the x argument in the fubction) is defined as an array (np.zeros(3) instead of 0.0)

# -----


# Gains
K = 10.0 * np.identity(3)

# Flag to detect if loop was interrupted
# interrupted = False

stop_event = threading.Event()
simulation_thread = None


def simulation_loop():
    global t, z, pd, dot_pd, q, qdot, tlog, plog, pdlog
    try:
        for i in range(50000):  # Equivalent to "while i < 50000 : "
            if stop_event.is_set():  # Early termination if stop_event is set
                print("Simulation stopped early.")
                break

            print("test")
            time.sleep(0.002)
            t_now = time.time()

            # Integrate time
            t = t + dt

            # Calculate DMP state derivatives (get z_dot, p_dot, pdot_dot)
            dz, dp, ddp = model.get_state_dot(z, pd, dot_pd)

            # Euler integration to update the states
            z += dz * dt
            pd += dp * dt
            dot_pd += ddp * dt

            # Get the joint values for the simulation
            q = q + qdot * dt

            # Get end-effector pose
            g = ur.fkine(q)
            R = np.array(g.R)
            p = np.array(g.t)

            # Get full jacobian
            J = np.array(ur.jacob0(q))

            # Get translational jacobian
            Jp = J[:3]

            # PseudoInverse
            JpInv = np.linalg.pinv(Jp)

            # Tracking control signal
            qdot = JpInv @ (dot_pd - K @ (p - pd))

            # Log data
            tlog = np.vstack((tlog, t))
            plog = np.vstack((plog, p))
            pdlog = np.vstack((pdlog, pd))  # Log the desired position over time

    except Exception as e:
        print(f"\nStopping robot due to KeyboardInterrupt: {e}")


# except KeyboardInterrupt:
#   print("\nStopping robot due to KeyboardInterrupt")
#   interrupted = True  # Set flag to indicate interruption


# Function to start the simulation
def start_simulation():
    global simulation_thread
    if simulation_thread is None or not simulation_thread.is_alive():
        simulation_thread = threading.Thread(target=simulation_loop, args=(rt))
        simulation_thread.start()
        print("Simulation started.")


def stop_simulation():
    if simulation_thread is not None and simulation_thread.is_alive():
        stop_event.set()
        simulation_thread.join()
        print("Simulation stopped.")


# Plotting the robot's position trajectory in 3D space
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# fig.suptitle('Robots Position Trajectory in 3D Space', fontsize=16)

# # Plotting actual and desired position trajectories in 3D
# ax.plot(pdlog[:, 0], pdlog[:, 1], pdlog[:, 2], color='b', linestyle='-', linewidth=1, label='Desired Position') # Desired position trajectory

# # Add labels, legend, and grid
# ax.set_xlabel('X Position (mm)', fontsize=14)
# ax.set_ylabel('Y Position (mm)', fontsize=14)
# ax.set_zlabel('Z Position (mm)', fontsize=14)
# ax.legend(loc="upper right", fontsize=12)
# ax.grid(True)

# plt.show()


# # Plotting the desired and actual position trajectories over time

# fig, ax = plt.subplots(figsize=(12, 8))
# fig.suptitle('Desired Position Trajectory (pd) Over Time', fontsize=16)

# # Colors for each position component
# colors = ['r', 'g', 'b']  # Red for x axis, green for y axis, blue for z axis

# # Plotting each component of pdlog against tlog
# for i in range(3):
#     ax.plot(tlog/2, plog[:, i], color=colors[i], linestyle='--', linewidth=1.5, label=f'Actual Position p{i+1}')  # Actual position
#     ax.plot(tlog/2, pdlog[:, i], color=colors[i], linestyle='-', linewidth=1, label=f'Desired Position pd{i+1}') # Desired position


# # Add labels, legend, and grid
# ax.set_xlabel('Time (s)', fontsize=14)
# ax.set_ylabel('Position (mm)', fontsize=14)
# ax.legend(loc="upper right", fontsize=12)
# ax.grid(True)

# plt.show()
