from deepbots.supervisor.controllers.supervisor_evolutionary import SupervisorEvolutionary
import numpy as np
from utilities import normalizeToRange, plotData

class CartpoleSupervisor(SupervisorEvolutionary):
    def __init__(self, model):
        super(CartpoleSupervisor, self).__init__(model)
        self.observationSpace = 4
        self.actionSpace = 2

        self.robot = None
        self.respawnRobot()
        self.poleEndpoint = self.getFromDef("POLE_ENDPOINT")
        self.messageReceived = None

        self.episodeCount = 0
        self.episodeLimit = 10000
        self.stepsPerEpisode = 200
        self.episodeScore = 0
        self.episodeScoreList = [] 

    def get_action(self, observation):
        q_values = self.model(observation)
        action = torch.argmax(q_values[0])

        return action

    def respawnRobot(self):
        if self.robot is not None:
            self.robot.remove()

            rootNode = self.getRoot()
            childrenField = rootNode.getField('children')
            childrenField.importMFNode(-2, "CartpoleRobot.wbo")

            self.robot = self.getFromDef("ROBOT")
            self.poleEndpoint = self.getFromDef("POLE_ENDPOINT")

    def get_observations(self):
        cartPosition = normalizeToRange(self.robot.getPosition()[2], -0.4, 0.4, -1.0, 1.0)
        cartVelocity = normalizeToRange(self.robot.getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True)
        endpointVelocity = normalizeToRange(self.poleEndpoint.getVelocity()[3], -1.5, 1.5, -1.0, 1.0, clip=True)

        self.messageReceived = self.handle_receiver()
        if self.messageReceived is not None:
            poleAngle = normalizeToRange(float(self.messageReceived[0]), -0.23, 0.23, -1.0, 1.0, clip=True)
        else:
            poleAngle = 0.0

        return [cartPosition, cartVelocity, poleAngle, endpointVelocity]

    def get_reward(self, action=None):
        return 1

    def is_done(self):
        if self.messageReceived is not None:
            poleAngle = round(float(self.messageReceived[0]), 2)
        else:
            poleAngle = 0.0
        
        if abs(poleAngle) > 0.26179938:
            return True

        if self.episodeScore > 195.0:
            return True

        cartPosition = round(self.robot.getPosition()[2], 2)

        if abs(cartPosition) > 0.39:
            return True
        
        return False

    def solved(self):
        if len(self.episodeScoreList) > 100:
            if np.mean(self.episodeScoreList[-100:]) > 195.0:
                return True

        return False

    def get_info(self):
        return None
            
        

    