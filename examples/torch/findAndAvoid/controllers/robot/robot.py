import numpy as np
from deepbots.robots.controllers.robot_emitter_receiver_csv import \
    RobotEmitterReceiverCSV


def normalize_to_range(value, min, max, newMin, newMax):
    value = float(value)
    min = float(min)
    max = float(max)
    newMin = float(newMin)
    newMax = float(newMax)
    return (newMax - newMin) / (max - min) * (value - max) + newMax


class FindTargetRobot(RobotEmitterReceiverCSV):
    def __init__(self, n_rangefinders):
        super(FindTargetRobot, self).__init__()
        self.setup_rangefinders(n_rangefinders)
        self.setup_motors()

    def create_message(self):
        message = []
        for rangefinder in self.rangefinders:
            message.append(rangefinder.getValue())
        return message

    def use_message_data(self, message):
        # Action 1 is gas
        gas = float(message[1])
        # Action 0 is turning
        wheel = float(message[0])

        # Normalzie gas from [-1, 1] to [0.3, 1.3] to make robot always move forward
        gas *= 4
        # Clip it
        # gas = np.clip(gas, -0.5, 4.0)

        # Clip turning rate to [-1, 1]
        wheel *= 2
        wheel = np.clip(wheel, -2, 2)

        # Apply gas to both motor speeds, add turning rate to one, subtract from other
        self.motorSpeeds[0] = gas + wheel
        self.motorSpeeds[1] = gas - wheel

        # Clip final motor speeds to [-4, 4] to be sure that motors get valid values
        self.motorSpeeds = np.clip(self.motorSpeeds, -4, 4)

        # Apply motor speeds
        self.leftMotor.setVelocity(self.motorSpeeds[0])
        self.rightMotor.setVelocity(self.motorSpeeds[1])

    def setup_rangefinders(self, n_rangefinders):
        # Sensor setup
        self.n_rangefinders = n_rangefinders
        self.rangefinders = []
        self.psNames = ['ps' + str(i) for i in range(self.n_rangefinders)
                        ]  # 'ps0', 'ps1',...,'ps7'

        for i in range(self.n_rangefinders):
            self.rangefinders.append(
                self.robot.getDistanceSensor(self.psNames[i]))
            self.rangefinders[i].enable(self.timestep)

    def setup_motors(self):
        # Motor setup
        self.leftMotor = self.robot.getMotor('left wheel motor')
        self.rightMotor = self.robot.getMotor('right wheel motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)

        self.motorSpeeds = [0.0, 0.0]


robot_controller = FindTargetRobot(8)
robot_controller.run()
