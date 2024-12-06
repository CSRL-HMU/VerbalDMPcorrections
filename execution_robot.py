import roboticstoolbox as rt
import numpy as np 
import rtde_receive
import rtde_control 
import math 
import keyboard  
import os 
import scipy.io
import time
from dmpR3 import *
from dmp import *
import pathlib
import scienceplots
from recTrainingData import *
from orientation import *
from CSRL_math import *
from dmpSO3 import *
from dmpSE3 import *

#define UR3e
rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")


#set initial configuration. Initial position of the robotic arm 
q0_c = np.array([-1.53928739, -2.47098269, -1.27327633, -1.57354273, 1.6510781, 11.8908723])

#Move leader to the initial folllower's pose 
rtde_c.moveJ(q0_c, 0.5, 0.5)

#get initial configuration 
q0 = np.array(rtde_r.getActualQ())


fileName = recTrainingData(rtde_c,rtde_r)


#set initial configuration. Initial position of the robotic arm 
q0_c = np.array([-1.53928739, -2.47098269, -1.27327633, -1.57354273, 1.6510781, 11.8908723])

#Move leader to the initial folllower's pose 
rtde_c.moveJ(q0_c, 0.5, 0.5)

#get initial configuration 
q0 = np.array(rtde_r.getActualQ())


#declare math pi 
pi = math.pi 

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
g0 = ur.fkine(q0)
R0 = np.array(g0.R)
p0 = np.array(g0.t)

Q0 = rot2quat(R0)

#Initial time 
t=0.0

#start logging 
plog = p0
pdlog = p0
tlog = t
Qlog = Q0
Qd_log = Q0

#initialize qdot
qdot = np.zeros(6)


# Loading the motion (data) and training the DMP model 
folderPath = pathlib.Path(__file__).parent.resolve()

# dataFile = folderPath / 'test_motion_demo.mat'  
dataFile = folderPath / fileName
data = scipy.io.loadmat(str(dataFile)) 

# Training the DMP model regararding the position
x_train = data['x']

wanted_x_train = x_train[1:, :]  #starts from the second row - till the end 


p_train =  np.array(np.transpose(wanted_x_train[:,0:3]))
Q_train =  np.array(np.transpose(wanted_x_train[:,3:7]))
t_train = np.array(list(range(p_train[1,:].size))) * dt


Q_target = Q_train[:,-1]

model = dmpSE3(N_in=20, T_in=t_train[-1]) 
model.train(dt, p_array=p_train, Q_array=Q_train, plotPerformance=True)


# setting the initial state of the DMP model
model.set_init_pose(p0, Q0)  #p
model.set_goal(gP_in= p_train[:,-1], gOr_in=Q_train[:,-1])

# print('p0=', p0)
pd = p0.copy()  # Initially, setting the desired position to the initial position p0
dot_pd = np.zeros(3)

ddp = np.zeros(3)
dp = np.zeros(3)
ddeo = np.zeros(3)
deo = np.zeros(3)

Q_desired = Q0.copy()  # Initially, setting the desired orientation to the initial orientation R0

eo = logError(Q_target, Q0)  # Initial orientation error
dot_eo = np.zeros(3)
ddot_eo = np.zeros(3)
omegad = np.zeros(3)
dot_omegad = np.zeros(3)

z = 0.0   #phase variable
dz = 0.0

    
# #### -------- -------- ####
# # Training the DMP model regarding the Orientation
# x_train = data['x']

# q_train = np.array(np.transpose( x_train ))

# t_train_q = np.array(list(range(q_train[1,:].size))) * dt

# model_q = dmpSO3(20, t_train_q[-1])
# model_q.train(dt, q_train, True)

# # setting the initial state of the DMP model reagrding the orientation
# model_q.set_init_state(R0)  #R

# pd_q = Q0.copy()  # Initially, setting the desired orientation to the initial orientation R0
# dot_pd_q = np.zeros(3)
# ddp_q = np.zeros(3)
# dp_q = np.zeros(3)
# z_q = 0.0   #phase variable
# dz_q = 0.0



####### Amplification test DMP regarding the Position #######

ampfactor = 1
tstart = t_train[-1]/3
tstop = 2*t_train[-1]/3
model.amplify_window(tstart, tstop, ampfactor)



# Gains
K = 4.0 * np.identity(6)


# User input to continue
continue_execution = input('continue...: ')


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
     
   
    # Calculate DMP state derivatives (get z_dot, p_dot, pdot_dot)
    # dz, dp, ddp = model.get_state_dot(z, pd, dot_pd)
    # get state dot
    dz, dp, ddp, deo, ddeo = model.get_state_dot(   z, 
                                                    pd,                                                                                      
                                                    dot_pd, 
                                                    eo,
                                                    dot_eo)
    
    
    Q_desired =  quatProduct( quatInv( quatExp( 0.5 * eo ) ) , Q_target )
    omegad = logDerivative2_AngleOmega(dot_eo, quatProduct(Q_desired,quatInv(Q_target)))
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
    eo_robot = logError(R, Q_desired)
    error_matrix = np.hstack((p - pd, eo_robot))        


    # tracking control signal
    qdot = Jinv @ (velocity_matrix - K @ error_matrix)
    
    # set joint speed
    rtde_c.speedJ(qdot, 5.0, dt)

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


#plot the training data
fig = plt.figure(figsize=(10, 8))
for i in range(3):
    axs = fig.add_axes([0.21, ((5-(i+3))/3)*0.8+0.2, 0.7, 0.25])
    axs.plot(t_train, p_train[i,:], 'r--', label='Training Data')
    axs.plot(tlog, plog[:,i], 'k', label='Executed Motion')
    axs.set_ylabel('p'+str(i+1))
    axs.set_xlabel('Time (s)')
    axs.legend()
    axs.grid(True)
plt.show()

# plot the quaternion training data
fig = plt.figure(figsize=(10, 8))
for i in range(4):
    axs = fig.add_subplot(4, 1, i+1)
    axs.plot(t_train, Q_train[i,:], 'r--', label='Q Training Data')
    axs.plot(tlog, Qlog[:,i], 'k', label='Q of Executed Motion')
    axs.set_ylabel('Q'+str(i+1))
    axs.set_xlabel('Time (s)')
    axs.legend()
    axs.grid(True)
plt.show()
    


