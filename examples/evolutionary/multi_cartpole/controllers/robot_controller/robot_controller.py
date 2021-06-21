from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV


class CartPoleRobot(RobotEmitterReceiverCSV):
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
        self.positionSensor = self.robot.getDevice("polePosSensor")
        self.positionSensor.enable(self.timestep)

        self.wheels = [None for _ in range(4)]
        self.setup_motors()

    def setup_motors(self):
        """
        This method initializes the four wheels, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.wheels[0] = self.robot.getDevice('wheel1')
        self.wheels[1] = self.robot.getDevice('wheel2')
        self.wheels[2] = self.robot.getDevice('wheel3')
        self.wheels[3] = self.robot.getDevice('wheel4')
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
        message = [self.robot.getName(), str(self.positionSensor.getValue())]
        return message

    def handle_receiver(self):
        """
        Modified handle_receiver from the basic implementation of deepbots.
        This one consumes all available messages in the queue during the step it is called.
        """
        while self.receiver.getQueueLength() > 0:
            # Receive and decode message from supervisor
            message = self.receiver.getData().decode("utf-8")
            # Convert string message into a list
            message = message.split(",")

            self.use_message_data(message)

            self.receiver.nextPacket()

    def use_message_data(self, message):
        """
        This method unpacks the supervisor's message, which contains the next action to be executed by the robot.
        In this case it contains an integer denoting the action, either 0 or 1, with 0 being forward and
        1 being backward movement. The corresponding motorSpeed value is applied to the wheels.

        :param message: The message the supervisor sent containing the next action.
        :type message: list of strings
        """
        if message[0] != "actions":
            return

        robotName = self.robot.getName()
        index = int(robotName[-1]) + 1
        action = int(message[index])

        assert action == 0 or action == 1, "CartPoleRobot controller got incorrect action value: " + str(action)

        if action == 0:
            motorSpeed = 5.0
        else:
            motorSpeed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motorSpeed)


# Create the robot controller object and run it
robot_controller = CartPoleRobot()
robot_controller.run()



'''from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV


class CartPoleRobot(RobotEmitterReceiverCSV):
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
        self.positionSensor = self.robot.getDevice("polePosSensor")
        self.positionSensor.enable(self.timestep)

        self.wheels = [None for _ in range(4)]
        self.setup_motors()

    def setup_motors(self):
        """
        This method initializes the four wheels, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.wheels[0] = self.robot.getDevice('wheel1')
        self.wheels[1] = self.robot.getDevice('wheel2')
        self.wheels[2] = self.robot.getDevice('wheel3')
        self.wheels[3] = self.robot.getDevice('wheel4')
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
        message = [self.robot.getName(), str(self.positionSensor.getValue())]
        return message

    def handle_receiver(self):
        """
        Modified handle_receiver from the basic implementation of deepbots.
        This one consumes all available messages in the queue during the step it is called.
        """
        while self.receiver.getQueueLength() > 0:
            # Receive and decode message from supervisor
            message = self.receiver.getData().decode("utf-8")
            # Convert string message into a list
            message = message.split(",")

            self.use_message_data(message)

            self.receiver.nextPacket()

    def use_message_data(self, message):
        """
        This method unpacks the supervisor's message, which contains the next action to be executed by the robot.
        In this case it contains an integer denoting the action, either 0 or 1, with 0 being forward and
        1 being backward movement. The corresponding motorSpeed value is applied to the wheels.

        :param message: The message the supervisor sent containing the next action.
        :type message: list of strings
        """
        if message[0] != "actions":
            return

        if self.robot.getName() == "robot1":
            action = int(message[1])
        else:
            action = int(message[2])

        assert action == 0 or action == 1, "CartPoleRobot controller got incorrect action value: " + str(action)

        if action == 0:
            motorSpeed = 5.0
        else:
            motorSpeed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motorSpeed)


# Create the robot controller object and run it
robot_controller = CartPoleRobot()
robot_controller.run()'''
