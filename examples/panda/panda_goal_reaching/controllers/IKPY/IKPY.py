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
armChain = Chain.from_urdf_file("./panda_with_bound.URDF")
# get position sensors
positionSensorList = Func.get_All_positionSensors(supervisor, timestep)
# get motors
motorList = Func.get_All_motors(supervisor)
# get a target
target = supervisor.getFromDef('TARGET'+str(np.random.randint(1, 10, 1)[0]))

rst_flag = 0
print("[IKPY start]")
while supervisor.step(timestep) != -1:

    targetPosition = ToArmCoord.convert(target.getPosition())

    # get values from position sensors
    psValue = Func.getValue(positionSensorList)
    psValue.append(0)
    psValue = np.array(psValue)
    psValue = np.insert(psValue, 0, 0)
    
    if rst_flag:
        # get another target
        target = supervisor.getFromDef('TARGET'+str(np.random.randint(1, 10, 1)[0]))
        # reset all motors
        done = Func.reset_All_motors(motorList, psValue)
        rst_flag = 0 if done else 1
    else:
        ikResults = armChain.inverse_kinematics(
            target_position=targetPosition, 
            initial_position=psValue)
        for i in range(7):
            motorList[i].setPosition(ikResults[i+1])
            motorList[i].setVelocity(1.0)
        prec = 0.0001
        err = np.absolute(np.array(psValue)-np.array(ikResults)) < prec
        rst_flag = 1 if np.all(err) else 0