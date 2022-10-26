from deepbots.supervisor.wrappers import KeyboardPrinter


class KeyboardControllerCartPole(KeyboardPrinter):
    def __init__(self, supervisor):
        super().__init__(supervisor)
        print("--------- Keyboard controls ---------")
        print("T: stop training and deploy agent for testing")
        print("R: reset world")
        print("(simulation window must be in focus)")
        print("------------------------------------")

    def step(self, action):
        """
        Overriding the default KeyboardPrinter step to add custom keyboard controls for CartPole problem.

        Pressing a button while the simulation window is in focus:

        "T" deploys the agent in testing mode and training is stopped.
        This can be useful if one wants to stop the simulation early and deploy the agent before the task is solved.

        "R" invokes the environment's reset method resetting the simulation to its initial state.
        """
        observation, reward, is_done, info = self.controller.step(action)
        key = self.keyboard.getKey()

        if key == ord("T") and not self.controller.test:
            self.controller.test = True
            print("Training will stop and agent will be deployed after episode end.")
        if key == ord("R"):
            print("User invoked reset method.")
            self.controller.reset()

        return observation, reward, is_done, info
