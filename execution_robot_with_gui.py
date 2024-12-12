import roboticstoolbox as rt
import numpy as np
import rtde_receive
import rtde_control
import math
import keyboard
import os
import scipy.io as scio
import scipy
import time
from dmpR3 import *
from dmp import *
import pathlib
import scienceplots
from orientation import *
from CSRL_math import *
from dmpSO3 import *
from dmpSE3 import *
import threading as thr
import time
import serial
import sys
import wave
import simpleaudio as sa
import serial
import numpy as np
import csrl_robotics
from robot_simulator import *
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


flagStop = False
fileName = "_"

# define UR3e
rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")


# Initialize serial connection
def initialize_serial():
    global ser
    try:
        serial_port = "/dev/tty.usbserial-A5069RR4"  # Adjust to your port
        baud_rate = 115200
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        # time.sleep(5)  # Wait for the port to initialize
        if ser.is_open:
            print("Serial port initialized and open.")
        else:
            ser.open()
            print("Serial port opened successfully.")
    except Exception as e:
        print(f"Failed to initialize serial port: {e}")
        ser = None  # Ensure no further use of ser if initialization fails


initialize_serial()


def inverse_kinematics(x, y, phi):
    # Internal definitions of links
    l1 = 3.9
    l2 = 3.4
    l3 = 2.5

    # Calculate the position of the wrist joint
    Xw = x - l3 * np.cos(phi)
    Yw = y - l3 * np.sin(phi)

    # Calculate theta2 and check if target is within reach
    C2 = (Xw**2 + Yw**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if C2 < -1 or C2 > 1:
        print(f"Target not reachable. C2 = {C2}")
        return

    theta2 = -np.arctan2(np.sqrt(1 - C2**2), C2)  # minus is to select  elbow up

    # Calculate theta1
    A = Xw * (l1 + l2 * np.cos(theta2)) + Yw * l2 * np.sin(theta2)
    B = Yw * (l1 + l2 * np.cos(theta2)) - Xw * l2 * np.sin(theta2)
    theta1 = np.arctan2(B, A)

    # Calculate theta3
    theta3 = phi - theta1 - theta2

    # Calculate the homogeneous transform for each joint
    T1 = csrl_robotics.RobotUtils.grotz(theta1)
    T2 = (
        T1
        @ csrl_robotics.RobotUtils.gtransl(l1, 0, 0)
        @ csrl_robotics.RobotUtils.grotz(theta2)
    )
    T3 = (
        T2
        @ csrl_robotics.RobotUtils.gtransl(l2, 0, 0)
        @ csrl_robotics.RobotUtils.grotz(theta3)
    )
    Tend = T3 @ csrl_robotics.RobotUtils.gtransl(l3, 0, 0)

    # Extract and return the position vector from Tend
    position = Tend[:3, 3]
    # print(f"position: {position}")
    Qs = np.array([theta1, theta2, theta3])
    # print(
    #     f"Q1: {np.rad2deg(theta1)}, Q2: {np.rad2deg(theta2)}, Q3: {np.rad2deg(theta3)}"
    # )
    return Qs


def send_commands(angle1, angle2, angle3):
    command = f"{angle1},{angle2},{angle3}\n"
    ser.write(command.encode())
    ser.flush()  # Ensure the command is sent
    # print(command)


def play_wav_file(file_path):
    try:
        # Open the .wav file
        with wave.open(file_path, "rb") as wav_file:
            # Read the parameters of the audio
            sample_rate = wav_file.getframerate()
            num_frames = int(sample_rate * 0.1)  # Calculate frames for 0.5 seconds

            # Read the first 0.5 seconds of data
            data = wav_file.readframes(num_frames)

        # Play the audio data
        play_obj = sa.play_buffer(data, 1, 2, sample_rate)
        play_obj.wait_done()  # Wait until playback finishes

    except Exception as e:
        print(f"Error: {e}")


def recTrainingData():
    global rtde_c, rtde_r, flagStop, fileName

    # get initial configuration
    q0 = np.array(rtde_r.getActualQ())

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

    # Control cycle
    dt = 0.002

    # get initial position
    q = np.array(rtde_r.getActualQ())
    g0 = ur.fkine(q)
    R0 = np.array(g0.R)
    p0 = np.array(g0.t)

    # Convert rotation matrix R to quaternion
    Q0 = rot2quat(R0)
    x0 = np.hstack((p0, Q0))

    # Init time
    t = 0.0

    # start logging
    plog = p0
    xlog = x0
    tlog = t

    demoname = input("Name of the demo: \n")

    rtde_c.teachMode()

    # make a sound for the human to start the demostration
    play_wav_file("/Users/elenikonstantinidou/Downloads/beep_sound.wav")

    for i in range(120000):
        # print(flagStop)
        if flagStop:
            print("Execution stopped by the user.")
            break

        t_start = rtde_c.initPeriod()

        # integrate time
        t = t + dt

        q = np.array(rtde_r.getActualQ())
        g = ur.fkine(q)
        R = np.array(g.R)
        p = np.array(g.t)

        # Convert rotation matrix R to quaternion
        Q = rot2quat(R)

        # Combining position and orientation in one state vector
        x = np.hstack((p, Q))

        tlog = np.vstack((tlog, t))
        plog = np.vstack((plog, p))
        xlog = np.vstack((xlog, x))

        rtde_c.waitPeriod(t_start)

    data = {"t": tlog, "p": plog, "x": xlog}
    fileName = demoname + "_demo.mat"
    scio.savemat(fileName, data)

    rtde_c.endTeachMode()

    return


# Function for opening the gripper
def open_gripper():
    try:
        if ser:
            ser.write(b"0\n")  # Command for opening the gripper
            ser.flush()
            print("Gripper is opening...")
            # time.sleep(1)  # Allow time for the gripper to open
        else:
            print("Serial connection is not initialized.")
    except Exception as e:
        print(f"An error occurred while opening the gripper: {e}")


# Function for closing the gripper
def close_gripper():
    try:
        if ser:
            ser.write(b"1\n")  # Command for closing the gripper
            ser.flush()
            print("Gripper is closing...")
            # time.sleep(1)  # Allow time for the gripper to close
        else:
            print("Serial connection is not initialized.")
    except Exception as e:
        print(f"An error occurred while closing the gripper: {e}")


#####################################################
# GUI for robot control
class RobotControlGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simulation Control GUI")
        self.setGeometry(100, 100, 400, 200)

        self.label = QLabel("Robot Control Buttons", self)
        self.label.setStyleSheet("color: white; font-size: 16px;")

        self.stop_button = QPushButton("Stop recording", self)
        self.stop_button.setStyleSheet(
            "background-color: #AF4C50; color: white; padding: 10px; border-radius: 5px;"
        )
        self.stop_button.clicked.connect(self.stop_recording)

        self.open_gripper_button = QPushButton("Open gripper", self)
        self.open_gripper_button.setStyleSheet(
            "background-color: #9966CC; color: white; padding: 10px; border-radius: 5px;"
        )
        self.open_gripper_button.clicked.connect(self.execute_sequence_opening)

        self.close_gripper_button = QPushButton("Close gripper", self)
        self.close_gripper_button.setStyleSheet(
            "background-color: #008080; color: white; padding: 10px; border-radius: 5px;"
        )
        self.close_gripper_button.clicked.connect(self.execute_sequence_closing)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.open_gripper_button)
        layout.addWidget(self.close_gripper_button)

        self.setLayout(layout)
        self.setStyleSheet("background-color: #121212;")

    def stop_recording(self):
        global flagStop
        flagStop = True
        print("Simulation stopped by the user.")

    def start_execution(self):
        global startExecution
        startExecution = True
        print("Execution is started.")

    def execute_sequence_opening(self):
        try:
            # Attempt to open the gripper
            if ser:
                ser.write(b"0\n")  # Command for opening the gripper
                ser.flush()
                print("Gripper is opening...")
            else:
                print("Serial connection is not initialized.")
                return

            # Proceed to execute the sequence after opening the gripper
            # print("Starting sequence...")
            x_start, y_start, phi_start = (
                5.3,
                2.5,
                (-np.pi / 3),
            )  # Starting position and orientation
            Q = inverse_kinematics(x_start, y_start, phi_start)
            send_commands(
                10 + np.rad2deg(Q[0]), 140 + np.rad2deg(Q[1]), 145 + np.rad2deg(Q[2])
            )
            # print("Sequence complete.")

        except Exception as e:
            print(f"An error occurred: {e}")

    def execute_sequence_closing(self):
        try:
            # Attempt to close the gripper
            if ser:
                ser.write(b"1\n")  # Command for closing the gripper
                ser.flush()

            print("Starting sequence...")
            x_end, y_end, phi_end = (
                6.5,
                -1.0,
                (-np.pi / 3),
            )  # Target position and orientation
            Q1 = inverse_kinematics(x_end, y_end, phi_end)
            send_commands(
                10 + np.rad2deg(Q1[0]), 140 + np.rad2deg(Q1[1]), 145 + np.rad2deg(Q1[2])
            )
            print("Sequence complete.")

        except Exception as e:
            print(f"An error occurred: {e}")


# set initial configuration. Initial position of the robotic arm
q0_c = np.array(
    [-1.53928739, -2.47098269, -1.27327633, -1.57354273, 1.6510781, 11.8908723]
)

# Move leader to the initial folllower's pose
rtde_c.moveJ(q0_c, 0.5, 0.5)

# get initial configuration
q0 = np.array(rtde_r.getActualQ())


# # Initialize the application
# app = QApplication(sys.argv)

# # Create the main window
# window = RobotControlGUI()
# window.show()

# # Start the thread only after the GUI is shown
# thread = thr.Thread(target=recTrainingData)
# thread.start()

# # Run the application event loop
# app.exec_()

# thread.join()


# Initialize the application
app = QApplication(sys.argv)

# Create the main window
window = RobotControlGUI()
window.show()

# Start the recTrainingData thread
training_thread = thr.Thread(target=recTrainingData, daemon=True)
training_thread.daemon = True  # Allow the thread to close when the main program exits
training_thread.start()


# Run the application event loop
app.exec_()

# Wait for the training thread to finish after the GUI loop exits
training_thread.join()

#####

# set initial configuration. Initial position of the robotic arm
q0_c = np.array(
    [-1.53928739, -2.47098269, -1.27327633, -1.57354273, 1.6510781, 11.8908723]
)

# Move leader to the initial folllower's pose
rtde_c.moveJ(q0_c, 0.5, 0.5)

# get initial configuration
q0 = np.array(rtde_r.getActualQ())

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

# Control cycle
dt = 0.002


# get initial end effector position
g0 = ur.fkine(q0)
R0 = np.array(g0.R)
p0 = np.array(g0.t)

Q0 = rot2quat(R0)

# Initial time
t = 0.0

# start logging
plog = p0
pdlog = p0
tlog = t
Qlog = Q0
Qd_log = Q0

# initialize qdot
qdot = np.zeros(6)


# Loading the motion (data) and training the DMP model
folderPath = pathlib.Path(__file__).parent.resolve()

# dataFile = folderPath / 'test_motion_demo.mat'
dataFile = folderPath / fileName
data = scipy.io.loadmat(str(dataFile))

# Training the DMP model regararding the position
x_train = data["x"]

wanted_x_train = x_train[1:, :]  # starts from the second row - till the end

p_train = np.array(np.transpose(wanted_x_train[:, 0:3]))
Q_train = np.array(np.transpose(wanted_x_train[:, 3:7]))
t_train = np.array(list(range(p_train[1, :].size))) * dt

Q_train = makeContinuous(Q_train)

Q_target = Q_train[:, -1]

model = dmpSE3(N_in=50, T_in=t_train[-1])
model.train(dt, p_array=p_train, Q_array=Q_train, plotPerformance=True)


# setting the initial state of the DMP model
model.set_init_pose(p0, Q0)  # p
model.set_goal(gP_in=p_train[:, -1], gOr_in=Q_train[:, -1])

# print('p0=', p0)
pd = p0.copy()  # Initially, setting the desired position to the initial position p0
dot_pd = np.zeros(3)

ddp = np.zeros(3)
dp = np.zeros(3)
ddeo = np.zeros(3)
deo = np.zeros(3)

Q_desired = (
    Q0.copy()
)  # Initially, setting the desired orientation to the initial orientation Q0
Q = Q0.copy()

eo = logError(Q_target, Q0)  # Initial orientation error
dot_eo = np.zeros(3)
ddot_eo = np.zeros(3)
omegad = np.zeros(3)
dot_omegad = np.zeros(3)

z = 0.0  # phase variable
dz = 0.0


####### Amplification #######

ampfactor = 1
tstart = t_train[-1] / 3
tstop = 2 * t_train[-1] / 3
model.amplify_window(tstart, tstop, ampfactor)


# Gains
K = 4.0 * np.identity(6)
K[-3:, -3:] = 5.0 * K[-3:, -3:]  # Orientation gains


# User input to continue
continue_execution = input("continue...: ")


i = 0
while t < t_train[-1]:
    i = i + 1

    t_now = time.time()
    t_start = rtde_c.initPeriod()

    # Integrate time
    t = t + dt

    # Euler integration to update the states
    z += dz * dt
    pd += dp * dt
    eo += deo * dt
    dot_eo += ddeo * dt
    dot_pd += ddp * dt

    # Get the desired position, velocity, acceleration, orientation, and angular velocity
    dz, dp, ddp, deo, ddeo = model.get_state_dot(z, pd, dot_pd, eo, dot_eo)

    Q_desired = quatProduct(quatInv(quatExp(0.5 * eo)), Q_target)
    omegad = logDerivative2_AngleOmega(
        dot_eo, quatProduct(Q_desired, quatInv(Q_target))
    )
    omegad = quat2rot(Q_target) @ omegad

    # Get the actual joint values
    q = np.array(rtde_r.getActualQ())

    # Get  end-efector pose
    g = ur.fkine(q)
    R = np.array(g.R)
    p = np.array(g.t)

    # get full jacobian
    J = np.array(ur.jacob0(q))

    # get translational jacobian
    # Jp = J[:3, :]

    # pseudoInverse
    Jinv = np.linalg.pinv(J)

    # velocity array
    velocity_matrix = np.hstack((dot_pd, omegad))

    # error matrix
    Q = rot2quatCont(R, Q)
    eo_robot = logError(Q, Q_desired)
    error_matrix = np.hstack((p - pd, eo_robot))

    # tracking control signal
    qdot = Jinv @ (velocity_matrix - K @ error_matrix)

    # set joint speed
    rtde_c.speedJ(qdot, 10.0, dt)

    # log data
    tlog = np.vstack((tlog, t))
    plog = np.vstack((plog, p))
    pdlog = np.vstack((pdlog, pd))

    Qlog = np.vstack((Qlog, rot2quat(R)))
    Qd_log = np.vstack((Qd_log, Q_desired))

    # synchronize
    rtde_c.waitPeriod(t_start)

# close control
rtde_c.speedStop()
rtde_c.stopScript()

############################################
# FUNCTIONS FOR THE GRIPPING


############################################

# PLOTTING THE FIGURES

# plot the training data
# fig = plt.figure(figsize=(10, 8))
# for i in range(3):
#     axs = fig.add_axes([0.21, ((5-(i+3))/3)*0.8+0.2, 0.7, 0.25])
#     axs.plot(t_train, p_train[i,:], 'r--', label='Training Data')
#     axs.plot(tlog, plog[:,i], 'k', label='Executed Motion')
#     axs.set_ylabel('p'+str(i+1))
#     axs.set_xlabel('Time (s)')
#     axs.legend()
#     axs.grid(True)
# plt.show()

# # plot the quaternion training data
# fig = plt.figure(figsize=(10, 8))
# for i in range(4):
#     axs = fig.add_subplot(4, 1, i+1)
#     axs.plot(t_train, Q_train[i,:], 'r--', label='Q Training Data')
#     axs.plot(tlog, Qlog[:,i], 'k', label='Q of Executed Motion')
#     axs.set_ylabel('Q'+str(i+1))
#     axs.set_xlabel('Time (s)')
#     axs.legend()
#     axs.grid(True)
# plt.show()
