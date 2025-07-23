"""IKPY controller."""
from controller import Supervisor
from ikpy.chain import Chain
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) # 
from ArmUtil import ToArmCoord, Func
import numpy as np

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

# IKPY Chain
arm_chain = Chain.from_urdf_file("./panda_with_bound.URDF")
# get position sensors
position_sensors = Func.get_all_position_sensors(supervisor, timestep)
# get motors
motors = Func.get_all_motors(supervisor)
# get a target
target = supervisor.getFromDef('TARGET'+str(np.random.randint(1, 10, 1)[0]))

rst_flag = 0
print("[IKPY start]")
while supervisor.step(timestep) != -1:

    target_position = ToArmCoord.convert(target.getPosition())

    # get values from position sensors
    ps_value = Func.get_value(position_sensors)
    ps_value.append(0)
    ps_value = np.array(ps_value)
    ps_value = np.insert(ps_value, 0, 0)
    
    if rst_flag:
        # get another target
        target = supervisor.getFromDef('TARGET'+str(np.random.randint(1, 10, 1)[0]))
        # reset all motors
        done = Func.reset_All_motors(motors, ps_value)
        rst_flag = 0 if done else 1
    else:
        ik_results = arm_chain.inverse_kinematics(
            target_position=target_position,
            initial_position=ps_value)
        for i in range(7):
            motors[i].setPosition(ik_results[i+1])
            motors[i].setVelocity(1.0)
        prec = 0.0001
        err = np.absolute(np.array(ps_value)-np.array(ik_results)) < prec
        rst_flag = 1 if np.all(err) else 0