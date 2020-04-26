from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
from numpy import argmax, isnan


class PitEscapeRobot(RobotEmitterReceiverCSV):
    """
    TODO
    """
    def __init__(self):
        """
        The constructor gets the Position Sensor reference and enables it and also initializes the wheels.
        """
        super().__init__()
        self.gyroSensor = self.robot.getGyro("body gyro")
        self.gyroSensor.enable(self.get_timestep())
        self.accelerometerSensor = self.robot.getAccelerometer("body accelerometer")
        self.accelerometerSensor.enable(self.get_timestep())

        # Max possible speed for the motor of the robot.
        self.maxSpeed = 8.72

        # Configuration of the main motor of the robot.
        self.pitchMotor = self.robot.getMotor("body pitch motor")
        self.yawMotor = self.robot.getMotor("body yaw motor")
        self.pitchMotor.setPosition(float('inf'))
        self.yawMotor.setPosition(float('inf'))
        self.pitchMotor.setVelocity(0.0)
        self.yawMotor.setVelocity(0.0)

    def create_message(self):
        """
        This method packs the robot's observation into a list of strings to be sent to the supervisor.
        # TODO fill this

        :return: list
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
        # TODO fill this

        :param message: list of strings, the message the supervisor sent, containing the next action
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

        self.pitchMotor.setVelocity(pitchSpeed)
        self.yawMotor.setVelocity(yawSpeed)


# Create the robot controller object and run it
robot_controller = PitEscapeRobot()
robot_controller.run()
