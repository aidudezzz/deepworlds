from deepbots.supervisor.wrappers.keyboard_printer import KeyboardPrinter
from controller import Keyboard


class KeyboardControllerCartPole(KeyboardPrinter):
    def __init__(self, supervisor):
        super().__init__(supervisor)

    def step(self, action):
        """
        Overriding the default KeyboardPrinter step to add custom keyboard controls for cartpole problem.

        Pressing a button while the simulation window is in focus:

        "T" deploys the agent in testing mode and training is stopped.
        This can be useful if one wants to stop the simulation early and deploy the agent before the task is solved.

        "R" invokes the environment's reset method resetting the simulation to its initial state.
        """
        observation, reward, isDone, info = self.controller.step(action)
        key = self.keyboard.getKey()

        if key == ord("T"):
            self.controller.test = True
            print("Training will stop and agent will be deployed after episode end.")
        if key == ord("R"):
            print("User invoked reset method")
            self.controller.reset()

        return observation, reward, isDone, info
