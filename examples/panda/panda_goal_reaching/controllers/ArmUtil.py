import numpy as np
class ToArmCoord:
	"""
	Convert from world coordinate (x, y, z)
	to arm coordinate (x, -z, y)
	"""
	@staticmethod
	def convert(world_coord):
		"""
		arg:
			world_coord: [x, y, z]
				An array of 3 containing the 3 world coordinate.
		"""
		return [world_coord[0], -world_coord[2], world_coord[1]]

class Func:
	@staticmethod
	def get_value(position_sensors):
		"""
		Get values from the position sensors
		"""
		ps_value = []
		for i in position_sensors:
			ps_value.append(i.getValue())
		return ps_value
	
	@staticmethod
	def get_all_motors(robot):
		"""
		Get 7 motors from the robot model
		"""
		names = ['motor' + str(i + 1) for i in range(7)]
		motors = []
		for i in names:
			motor = robot.getDevice(i)	 # Get the motor handle #position_sensor1
			motor.setPosition(float('inf'))  # Set starting position
			motor.setVelocity(0.0)  # Zero out starting velocity
			motors.append(motor)
		return motors
		
	@staticmethod
	def get_all_position_sensors(robot, timestep):
		"""
		Get 7 position sensors from the robot model
		"""
		position_sensors = []
		for i in range(7):
			name = 'positionSensor' + str(i+1)
			position_sensor = robot.getDevice(name)
			position_sensor.enable(timestep)
			position_sensors.append(position_sensor)
		return position_sensors
	
	@staticmethod
	def reset_all_motors(motors, ps_value):
		"""
		Reset 7 motors on the robot model
		"""
		reset_value = [0.0, 0.0, 0.0, -0.0698, 0.0, 0.0, 0.0]

		for i in range(len(motors)):
			motors[i].setPosition(reset_value[i])  # Set starting position
			motors[i].setVelocity(1.0)  # Zero out starting velocity
		
		prec = 0.0001
		err = np.absolute(np.array(ps_value[1:8])-np.array(reset_value)) < prec
		return 1 if np.all(err) else 0