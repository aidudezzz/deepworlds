import numpy as np
from deepbots.robots import CSVRobot


def normalize_to_range(value, min, max, new_min, new_max):
    value = float(value)
    min = float(min)
    max = float(max)
    new_min = float(new_min)
    new_max = float(new_max)
    return (new_max - new_min) / (max - min) * (value - max) + new_max


class FindTargetRobot(CSVRobot):
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
        self.motor_speeds[0] = gas + wheel
        self.motor_speeds[1] = gas - wheel

        # Clip final motor speeds to [-4, 4] to be sure that motors get valid values
        self.motor_speeds = np.clip(self.motor_speeds, 0, 6)

        # Apply motor speeds
        self._set_velocity(self.motor_speeds[0], self.motor_speeds[1])

    def setup_rangefinders(self, n_rangefinders):
        # Sensor setup
        self.n_rangefinders = n_rangefinders
        self.rangefinders = []
        self.ps_names = ['ps' + str(i) for i in range(self.n_rangefinders)
                        ]  # 'ps0', 'ps1',...,'ps7'

        for i in range(self.n_rangefinders):
            self.rangefinders.append(self.getDevice(self.ps_names[i]))
            self.rangefinders[i].enable(self.timestep)

    def setup_motors(self):
        # Motor setup
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self._set_velocity(0.0, 0.0)
        self.motor_speeds = [0.0, 0.0]
    
    def _set_velocity(self, v_left, v_right):
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)


robot_controller = FindTargetRobot(8)
robot_controller.run()
