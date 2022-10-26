from deepbots.robots import CSVRobot


class CartPoleRobot(CSVRobot):
    """
    CartPole robot has 4 wheels and pole connected by an unactuated hinge to its body.
    The hinge contains a Position Sensor device to measure the angle from vertical needed in the observation.
    Hinge: https://cyberbotics.com/doc/reference/hingejoint
    Position Sensor: https://cyberbotics.com/doc/reference/positionsensor
    """

    def __init__(self):
        """
        The constructor gets the Position Sensor reference and enables it and also initializes the wheels.
        """
        super().__init__()
        self.position_sensor = self.getDevice("polePosSensor")
        self.position_sensor.enable(self.timestep)

        self.wheels = [None for _ in range(4)]
        self.setup_motors()

    def setup_motors(self):
        """
        This method initializes the four wheels, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.wheels[0] = self.getDevice('wheel1')
        self.wheels[1] = self.getDevice('wheel2')
        self.wheels[2] = self.getDevice('wheel3')
        self.wheels[3] = self.getDevice('wheel4')
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(0.0)

    def create_message(self):
        """
        This method packs the robot's observation into a list of strings to be sent to the supervisor.
        The message contains only the Position Sensor value, ie. the angle from vertical position in radians.
        From Webots documentation:
        'The getValue function returns the most recent value measured by the specified position sensor. Depending on
        the type, it will return a value in radians (angular position sensor) or in meters (linear position sensor).'

        :return: A list of strings with the robot's observations.
        :rtype: list
        """
        message = [str(self.position_sensor.getValue())]
        return message

    def use_message_data(self, message):
        """
        This method unpacks the supervisor's message, which contains the next action to be executed by the robot.
        The message contains a float value that is applied on all wheels as velocity.

        :param message: The message the supervisor sent containing the next action.
        :type message: list of strings
        """
        motor_speed = float(message[0]) * 5.0  # Scale from [-1.0, 1.0] to [-5.0, 5.0]

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motor_speed)


# Create the robot controller object and run it
robot_controller = CartPoleRobot()
robot_controller.run()
