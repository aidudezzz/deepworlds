import numpy as np

class RobotFunc(object):
    
    def getMotorNames(name_type='all'):
        if name_type == 'all':
            return ['Head', 'LeftAnkle', 'LeftArm', 'LeftCrus', 
                    'LeftElbow', 'LeftFemur','LeftFemurHead1', 
                    'LeftFemurHead2', 'LeftFoot', 'LeftForearm',
                    'LeftShoulder','RightAnkle', 'RightArm', 
                    'RightCrus', 'RightElbow', 'RightFemur',
                    'RightFemurHead1', 'RightFemurHead2', 
                    'RightFoot', 'RightForearm','RightShoulder',
                    'Waist']
        else:
            return ['LeftAnkle', 'LeftCrus', 'LeftFemur',
                    'LeftFoot','RightAnkle', 'RightCrus','RightFemur',
                    'RightFoot','Waist']
                
                
    def getAllMotors(robot):
        """
        Get 17 motors from the robot model
        """
        
        motorNames = RobotFunc.getMotorNames('all')
        
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

