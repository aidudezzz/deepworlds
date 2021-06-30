import numpy as np
class ToArmCoord:
	"""
	Convert from world coordinate (x, y, z)
	to arm coordinate (x, -z, y)
	"""
	@staticmethod
	def convert(worldCoord):
		"""
		arg:
			worldCoord: [x, y, z]
				An array of 3 containing the 3 world coordinate.
		"""
		return [worldCoord[0], -worldCoord[2], worldCoord[1]]

class Func:
	@staticmethod
	def getValue(positionSensorList):
		"""
		Get values from the position sensors
		"""
		psValue = []
		for i in positionSensorList:
			psValue.append(i.getValue())
		return psValue
	
	@staticmethod
	def get_All_motors(robot):
		"""
		Get 7 motors from the robot model
		"""
		nameList = ['motor' + str(i + 1) for i in range(7)]
		motorList = []
		for i in nameList:
			motor = robot.getDevice(i)	 # Get the motor handle #positionSensor1
			motor.setPosition(float('inf'))  # Set starting position
			motor.setVelocity(0.0)  # Zero out starting velocity
			motorList.append(motor)
		return motorList
		
	@staticmethod
	def get_All_positionSensors(robot, timestep):
		"""
		Get 7 position sensors from the robot model
		"""
		positionSensorList = []
		for i in range(7):
			positionSensorName = 'positionSensor' + str(i+1)
			positionSensor = robot.getDevice(positionSensorName)
			positionSensor.enable(timestep)
			positionSensorList.append(positionSensor)
		return positionSensorList
	
	@staticmethod
	def reset_All_motors(motorList, psValue):
		"""
		Reset 7 motors on the robot model
		"""
		resetValue = [0.0, 0.0, 0.0, -0.0698, 0.0, 0.0, 0.0]

		for i in range(len(motorList)):
			motorList[i].setPosition(resetValue[i])  # Set starting position
			motorList[i].setVelocity(1.0)  # Zero out starting velocity
		
		prec = 0.0001
		err = np.absolute(np.array(psValue[1:8])-np.array(resetValue)) < prec
		return 1 if np.all(err) else 0