
import roboticstoolbox as rt
import numpy as np
import rtde_receive
import rtde_control
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math


# Declare math pi
pi = math.pi

ip_robot = "192.168.1.60"     # for UR3e
# ip_robot = "192.168.1.100"      # for UR5e

# Define robot
rtde_c = rtde_control.RTDEControlInterface(ip_robot)
rtde_r = rtde_receive.RTDEReceiveInterface(ip_robot)

q = rtde_r.getActualQ()
print('q (rad) = ', q)
print('q (deg) = ', np.array(q)*180/pi)

#define the robot with its DH parameters 
ur = rt.DHRobot([
    rt.RevoluteDH(d = 0.15185, alpha = pi/2),
    rt.RevoluteDH(a = -0.24355),
    rt.RevoluteDH(a = -0.2132),
    rt.RevoluteDH(d = 0.13105, alpha = pi/2),
    rt.RevoluteDH(d = 0.08535, alpha = -pi/2),
    rt.RevoluteDH(d = 0.0921)
], name='UR3e')

#Control cycle 
dt=0.002

#get initial end effector position 
g0 = ur.fkine(np.array(q))
R0 = np.array(g0.R)
p0 = np.array(g0.t)

print('g0 = ', g0)
print('R = ', R0)
print('p = ', p0)

# stop the robot
rtde_c.speedStop()
rtde_c.stopScript()










