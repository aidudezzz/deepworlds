import numpy as np

class RobotFunc(object):
    
    def getMotorNames(name_type='all'):
        if name_type == 'all':
            return ['HeadPitch','HeadYaw', 'LAnklePitch', 'LAnkleRoll', 'LElbowRoll',
            'LElbowYaw', 'LHipPitch', 'LHipRoll', 'LHipYawPitch', 'LKneePitch', 'LShoulderPitch',
            'LShoulderRoll','LWristYaw', 'RAnklePitch', 'RAnkleRoll', 'RElbowRoll', 'RElbowYaw',
            'RHipPitch', 'RHipRoll', 'RHipYawPitch', 'RKneePitch', 'RShoulderPitch', 'RShoulderRoll',
            'RWristYaw']
        elif name_type=='legs':
            return ['LAnklePitch', 'LHipPitch', 'LKneePitch',
                    'RAnklePitch', 'RHipPitch', 'RKneePitch',
                    'LShoulderPitch',"RShoulderPitch"]
                    
    def getAllMotors(robot):
        """
        Get 24 (all) or 8 (leg, hands) motors from the robot model
        """
        motorNames = RobotFunc.getMotorNames('legs')
        
        motorList = []
        for motorName in motorNames:
            motor = robot.getDevice(motorName)	 # Get the motor handle #positionSensor1
            motor.setPosition(float('inf'))  # Set starting position
            motor.setVelocity(0.0)  # Zero out starting velocity
            motorList.append(motor)
        return motorList
        
    def normalizeToRange(value, minVal, maxVal, newMin, newMax, clip=False):
        """
        Normalizes value to a specified new range by supplying the current range.
        :param value: value to be normalized
        :type value: float
        :param minVal: value's min value, value ∈ [minVal, maxVal]
        :type minVal: float
        :param maxVal: value's max value, value ∈ [minVal, maxVal]
        :type maxVal: float
        :param newMin: normalized range min value
        :type newMin: float
        :param newMax: normalized range max value
        :type newMax: float
        :param clip: whether to clip normalized value to new range or not, defaults to False
        :type clip: bool, optional
        :return: normalized value ∈ [newMin, newMax]
        :rtype: float
        """
        
        value = float(value)
        minVal = float(minVal)
        maxVal = float(maxVal)
        newMin = float(newMin)
        newMax = float(newMax)

        if clip:
            return np.clip((newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax, newMin, newMax)
        else:
            return (newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax