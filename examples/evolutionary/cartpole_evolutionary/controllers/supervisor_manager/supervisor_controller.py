from deepbots.supervisor.controllers.supervisor_evolutionary import SupervisorEvolutionary
from controller import Supervisor
import numpy as np
import torch.nn as nn
import torch
from utilities import normalizeToRange

class CartpoleSupervisor(SupervisorEvolutionary):
    def __init__(self, model):
        super().__init__(model)
        self.observationSpace = 4
        self.actionSpace = 2

        self.robot = self.getFromDef("ROBOT")
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

        return [action.item()]

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

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        return [0.0 for _ in range(self.observationSpace)]

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

    def reset(self):
        """
        Used to reset the world to an initial state.
        Default, problem-agnostic, implementation of reset method,
        using Webots-provided methods.
        *Note that this works properly only with Webots versions >R2020b
        and must be overridden with a custom reset method when using
        earlier versions. It is backwards compatible due to the fact
        that the new reset method gets overridden by whatever the user
        has previously implemented, so an old supervisor can be migrated
        easily to use this class.
        :return: default observation provided by get_default_observation()
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        self.episodeScore = 0

        # print("Before Supervisor Receiver0: ",
        #       self.communication[0]['receiver'].getQueueLength())
        # print("Before Supervisor Receiver1: ",
        #       self.communication[1]['receiver'].getQueueLength())

        
        while self.receiver.getQueueLength() > 0:
            self.receiver.nextPacket()

        # print("After Supervisor Receiver0: ",
        #       self.communication[0]['receiver'].getQueueLength())
        # print("After Supervisor Receiver1: ",
        #       self.communication[1]['receiver'].getQueueLength())

        return self.get_default_observation()