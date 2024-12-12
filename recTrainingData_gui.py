import roboticstoolbox as rt
import numpy as np
import rtde_receive
import rtde_control
import math
import os
import scipy.io as scio
import wave
import simpleaudio as sa
from orientation import *
from CSRL_math import *
from dmpSO3 import *
from dmpSE3 import *
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)
import sys
import threading as thr


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


def recTrainingData(rtde_c, rtde_r):

    global flagStop

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
        print(flagStop)
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

    return fileName
