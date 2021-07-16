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

        # Mapping gas from [-1, 1] to [0, 4] to make robot always move forward
        gas = (gas+1)*2
        gas = np.clip(gas, 0, 4.0)

        # Mapping turning rate from [-1, 1] to [-2, 2]
        wheel *= 2
        wheel = np.clip(wheel, -2, 2)

        # Apply gas to both motor speeds, add turning rate to one, subtract from other
        self.motorSpeeds[0] = gas + wheel
        self.motorSpeeds[1] = gas - wheel

        # Clip final motor speeds to [-4, 4] to be sure that motors get valid values
        self.motorSpeeds = np.clip(self.motorSpeeds, 0, 6)

        # Apply motor speeds
        self._setVelocity(self.motorSpeeds[0], self.motorSpeeds[1])

    def setup_rangefinders(self, n_rangefinders):
        # Sensor setup
        self.n_rangefinders = n_rangefinders
        self.rangefinders = []
        self.psNames = ['ps' + str(i) for i in range(self.n_rangefinders)
                        ]  # 'ps0', 'ps1',...,'ps7'

        for i in range(self.n_rangefinders):
            self.rangefinders.append(self.robot.getDevice(self.psNames[i]))
            self.rangefinders[i].enable(self.timestep)

    def setup_motors(self):
        # Motor setup
        self.leftMotor = self.robot.getDevice('left wheel motor')
        self.rightMotor = self.robot.getDevice('right wheel motor')
        self._setVelocity(0.0, 0.0)
        self.motorSpeeds = [0.0, 0.0]
    
    def _setVelocity(self, v1, v2):
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(v1)
        self.rightMotor.setVelocity(v2)


robot_controller = FindTargetRobot(8)
robot_controller.run()
