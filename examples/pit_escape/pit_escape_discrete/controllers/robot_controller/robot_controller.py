from deepbots.robots import CSVRobot
from numpy import isnan


class PitEscapeRobot(CSVRobot):
    """
    The BB-8 robot consists of a big ball which acts as a wheel, that moves it around.
    With this controller, it has 2 possible actions, pitch and yaw. It also uses a
    gyro sensor and an accelerometer.
    BB-8 doc: https://cyberbotics.com/doc/guide/bb8
    """

    def __init__(self):
        """
        The constructor initializes the sensors (gyro and accelerometer) and the motors (pitch and yaw).
        """
        super().__init__()
        # Set up sensors
        self.gyro_sensor = self.getDevice("body gyro")
        self.gyro_sensor.enable(self.timestep)
        self.accelerometer_sensor = self.getDevice("body accelerometer")
        self.accelerometer_sensor.enable(self.timestep)

        # Max possible speed for the motor
        self.max_speed = 8.72

        # Configuration of the main motors of the robot
        self.pitch_motor = self.getDevice("body pitch motor")
        self.yaw_motor = self.getDevice("body yaw motor")
        self.pitch_motor.setPosition(float('inf'))
        self.yaw_motor.setPosition(float('inf'))
        self.pitch_motor.setVelocity(0.0)
        self.yaw_motor.setVelocity(0.0)

    def create_message(self):
        """
        This method packs the robot's observation into a list of strings to be sent to the supervisor.
        Sometimes the two sensors return NaNs. In those cases zero vectors are returned.

        :return: A list of strings with the robot's observations.
        :rtype: list
        """
        gyro_values = self.gyro_sensor.getValues()
        accelerometer_values = self.accelerometer_sensor.getValues()
        if True in isnan(gyro_values):
            # A nan is found in the sensor values
            message = ["0" for _ in range(3)]
        else:
            message = [str(gyro_values[i]) for i in range(len(gyro_values))]
        if True in isnan(accelerometer_values):
            # A nan is found in the sensor values
            message.extend(["0" for _ in range(3)])
        else:
            message.extend([str(accelerometer_values[i]) for i in range(len(accelerometer_values))])

        return message

    def use_message_data(self, message):
        """
        This method unpacks the supervisor's message, which contains the next action to be executed by the robot.
        After the action is converted into an integer, it can take the values 0, 1, 2 and 3, which in pairs
        correspond to pitch and yaw. Based on the action, the max speed is set on the appropriate motor.

        :param message: The message the supervisor sent containing the next action.
        :type message: list of strings
        """
        pitch_speed = 0
        yaw_speed = 0
        action = int(message[0])
        # print("new action:", end="")
        if action == 0:
            # print("pitch +")
            pitch_speed = self.max_speed
        elif action == 1:
            # print("pitch -")
            pitch_speed = -self.max_speed
        elif action == 2:
            # print("yaw +")
            yaw_speed = self.max_speed
        elif action == 3:
            # print("yaw -")
            yaw_speed = -self.max_speed

        self.pitch_motor.setPosition(float('inf'))
        self.yaw_motor.setPosition(float('inf'))
        self.pitch_motor.setVelocity(pitch_speed)
        self.yaw_motor.setVelocity(yaw_speed)


# Create the robot controller object and run it
robot_controller = PitEscapeRobot()
robot_controller.run()
