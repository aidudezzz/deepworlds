from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
from numpy import isnan


class PitEscapeRobot(RobotEmitterReceiverCSV):
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
        self.gyroSensor = self.robot.getDevice("body gyro")
        self.gyroSensor.enable(self.timestep)
        self.accelerometerSensor = self.robot.getDevice("body accelerometer")
        self.accelerometerSensor.enable(self.timestep)

        # Max possible speed for the motor
        self.maxSpeed = 8.72

        # Configuration of the main motors of the robot
        self.pitchMotor = self.robot.getDevice("body pitch motor")
        self.yawMotor = self.robot.getDevice("body yaw motor")
        self.pitchMotor.setPosition(float('inf'))
        self.yawMotor.setPosition(float('inf'))
        self.pitchMotor.setVelocity(0.0)
        self.yawMotor.setVelocity(0.0)

    def create_message(self):
        """
        This method packs the robot's observation into a list of strings to be sent to the supervisor.
        Some times the two sensors return NaNs. In those cases zero vectors are returned.

        :return: A list of strings with the robot's observations.
        :rtype: list
        """
        gyroValues = self.gyroSensor.getValues()
        accelerometerValues = self.accelerometerSensor.getValues()
        if True in isnan(gyroValues):
            # A nan is found in the sensor values
            message = ["0" for _ in range(3)]
        else:
            message = [str(gyroValues[i]) for i in range(len(gyroValues))]
        if True in isnan(accelerometerValues):
            # A nan is found in the sensor values
            message.extend(["0" for _ in range(3)])
        else:
            message.extend([str(accelerometerValues[i]) for i in range(len(accelerometerValues))])

        return message

    def use_message_data(self, message):
        """
        This method unpacks the supervisor's message, which contains the next action to be executed by the robot.
        After the action is converted into an integer, it can take the values 0, 1, 2 and 3, which in pairs
        correspond to pitch and yaw. Based on the action, the max speed is set on the appropriate motor.

        :param message: The message the supervisor sent containing the next action.
        :type message: list of strings
        """
        pitchSpeed = 0
        yawSpeed = 0
        action = int(message[0])
        # print("new action:", end="")
        if action == 0:
            # print("pitch +")
            pitchSpeed = self.maxSpeed
        elif action == 1:
            # print("pitch -")
            pitchSpeed = -self.maxSpeed
        elif action == 2:
            # print("yaw +")
            yawSpeed = self.maxSpeed
        elif action == 3:
            # print("yaw -")
            yawSpeed = -self.maxSpeed

        self.pitchMotor.setPosition(float('inf'))
        self.yawMotor.setPosition(float('inf'))
        self.pitchMotor.setVelocity(pitchSpeed)
        self.yawMotor.setVelocity(yawSpeed)


# Create the robot controller object and run it
robot_controller = PitEscapeRobot()
robot_controller.run()
