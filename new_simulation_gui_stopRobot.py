import tkinter as tk
import threading
import numpy as np
import math
import time
import scipy.io
from dmpR3 import *
import roboticstoolbox as rt
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set initial configuration. Initial position of the robotic arm
q0_c = np.array([-1.53928739, -2.47098269, -1.27327633, -1.57354273, 1.6510781, 11.8908723])
q0 = q0_c.copy()
q = q0.copy()

pi = math.pi

# Define the robot with its DH parameters
ur = rt.DHRobot([
    rt.RevoluteDH(d=0.15185, alpha=pi/2),
    rt.RevoluteDH(a=-0.24355),
    rt.RevoluteDH(a=-0.2132),
    rt.RevoluteDH(d=0.13105, alpha=pi/2),
    rt.RevoluteDH(d=0.08535, alpha=-pi/2),
    rt.RevoluteDH(d=0.0921)
], name='UR3e')

dt = 0.002  # Control cycle time

# Load data and train the DMP model
folderPath = pathlib.Path(__file__).parent.resolve()
dataFile = folderPath / 'test_motion_demo.mat'
data = scipy.io.loadmat(str(dataFile))
x_train = data['p']
p_train = np.array(np.transpose(x_train[1:, :]))
t_train = np.array(list(range(p_train[1, :].size))) * dt

model = dmpR3(20, t_train[-1])
model.train(dt, p_train, False)
p0 = ur.fkine(q0).t
model.set_init_state(p0)

# Simulation state variables
pd = p0.copy()
dot_pd = np.zeros(3)
z = 0.0
qdot = np.zeros(6)
K = 10.0 * np.identity(3)

# Logging
t = 0.0
tlog = np.array([t])
plog = np.array([p0])
pdlog = np.array([pd])

# Event to stop the simulation
stop_event = threading.Event()

def simulation_loop():
    global t, z, pd, dot_pd, q, qdot, tlog, plog, pdlog
    try:
        for i in range(50000):  # Loop will iterate 50000 times
            if stop_event.is_set():  # Early termination if stop_event is set
                print("Simulation stopped early.")
                break

            print('test')
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

            # get the joint values for the simulation 
            q = q + qdot * dt

            # Get end-effector pose
            g = ur.fkine(q)
            R = np.array(g.R)
            p = np.array(g.t)

            # get full jacobian
            J = np.array(ur.jacob0(q))

            # get translational jacobian
            Jp = J[:3]

            # pseudoInverse
            JpInv = np.linalg.pinv(Jp)

            # tracking control signal
            qdot = JpInv @ (dot_pd - K @ (p - pd))

            # log data
            tlog = np.vstack((tlog, t))
            plog = np.vstack((plog, p))
            pdlog = np.vstack((pdlog, pd))

        print("Simulation completed after 50000 iterations.")
    except Exception as e:
        print(f"Error in simulation: {e}")
        
        
# Function to start the simulation
def start_simulation():
    stop_event.clear()
    threading.Thread(target=simulation_loop, daemon=True).start()
    print("Simulation started.")

# Function to stop the simulation
def stop_simulation():
    stop_event.set()
    print("Stop signal sent.")

# GUI for control
root = tk.Tk()
root.title("Robot Simulation Control")

start_button = tk.Button(root, text="Start Simulation", command=start_simulation)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Simulation", command=stop_simulation)
stop_button.pack(pady=10)

root.mainloop()

# Plotting (can be done after the GUI is closed)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(pdlog[:, 0], pdlog[:, 1], pdlog[:, 2], label='Desired Position')
ax.plot(plog[:, 0], plog[:, 1], plog[:, 2], label='Actual Position')
ax.set_xlabel("X Position (mm)")
ax.set_ylabel("Y Position (mm)")
ax.set_zlabel("Z Position (mm)")
ax.legend()
plt.show()