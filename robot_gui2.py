import tkinter as tk
import threading
import time
import serial
import numpy as np
import csrl_robotics
from robot_simulator import *
from orientation import *
from CSRL_math import *
from dmpSO3 import *
from dmpSE3 import *
from robot_simulator import *


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


# Functions for executing the full sequence
def execute_sequence_opening():
    try:
        print("Starting sequence...")
        open_gripper()
        x_start, y_start, phi_start = (
            5.3,
            2.5,
            (-np.pi / 3),
        )  # Starting position and orientation
        Q = inverse_kinematics(x_start, y_start, phi_start)
        send_commands(
            10 + np.rad2deg(Q[0]), 140 + np.rad2deg(Q[1]), 145 + np.rad2deg(Q[2])
        )
        # move_to_initial_position()
        print("Sequence complete.")
    except Exception as e:
        print(f"An error occurred during the sequence: {e}")


def execute_sequence_closing():
    try:
        print("Starting sequence...")
        close_gripper()  # Step 2: Close the gripper
        x_end, y_end, phi_end = (
            6.5,
            -1.0,
            (-np.pi / 3),
        )  # Target position and orientation
        Q1 = inverse_kinematics(x_end, y_end, phi_end)
        send_commands(
            10 + np.rad2deg(Q1[0]), 140 + np.rad2deg(Q1[1]), 145 + np.rad2deg(Q1[2])
        )
        # move_to_final_position()  # Step 1: Move to initial position
        print("Sequence complete.")
    except Exception as e:
        print(f"An error occurred during the sequence: {e}")


############################################


# GUI class to manage the interface
class RobotControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gripper Control")
        self.root.geometry("300x300")

        # Simulation Controls
        self.simulation_label = tk.Label(
            self.root, text="Simulation Controls", font=("Helvetica", 14, "bold")
        )
        self.simulation_label.pack(pady=10)

        self.start_simulation_button = tk.Button(
            self.root,
            text="Start Simulation",
            command=self.run_in_thread(start_simulation),
        )
        self.start_simulation_button.pack(pady=10)

        self.stop_simulation_button = tk.Button(
            self.root,
            text="Stop Simulation",
            command=self.run_in_thread(stop_simulation),
        )
        self.stop_simulation_button.pack(pady=10)

        # Gripper Controls
        self.gripper_label = tk.Label(
            self.root, text="Gripper Controls", font=("Helvetica", 14, "bold")
        )
        self.gripper_label.pack(pady=10)

        # Execute Sequence button for opening gripper
        self.execute_open_button = tk.Button(
            self.root,
            text="Open Gripper",
            command=self.run_in_thread(execute_sequence_opening),
        )
        self.execute_open_button.pack(pady=10)

        # Execute Sequence button for closing gripper
        self.execute_close_button = tk.Button(
            self.root,
            text="Close Gripper",
            command=self.run_in_thread(execute_sequence_closing),
        )
        self.execute_close_button.pack(pady=10)

    def run_in_thread(self, target_func):
        def wrapper():
            thread = threading.Thread(target=target_func)
            thread.daemon = True
            thread.start()

        return wrapper


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
    print(f"position: {position}")
    Qs = np.array([theta1, theta2, theta3])
    print(
        f"Q1: {np.rad2deg(theta1)}, Q2: {np.rad2deg(theta2)}, Q3: {np.rad2deg(theta3)}"
    )
    return Qs


def send_commands(angle1, angle2, angle3):
    command = f"{angle1},{angle2},{angle3}\n"
    ser.write(command.encode())
    ser.flush()  # Ensure the command is sent
    print(command)


# Main function to run the GUI
def main():
    root = tk.Tk()
    app = RobotControlGUI(root)
    root.mainloop()


# Run the GUI
if __name__ == "__main__":
    initialize_serial()
    main()
