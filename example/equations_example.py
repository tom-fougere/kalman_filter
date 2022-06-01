import numpy as np

from extended.equations import KalmanEquation


class ExampleEquation(KalmanEquation):
    def __init__(self, gravity, Te):
        super().__init__()

        self.gravity = gravity
        self.Te = Te

    def state_equation(self, states, commands=None):
        """
        State Equation
        """

        state_matrix = [
            states[0],
            self.Te * states[0] + states[1],
            states[2],
        ]
        return np.asarray(state_matrix)

    def jacobian_state_equation(self, states, commands=None):
        """
        Jacobian matrix of state Equation
        """
        jac = [
            [1, 0, 0],
            [self.Te, 1, 0],
            [0, 0, 1],
        ]
        return np.asarray(jac)

    def obs_equation(self, states):
        """
        Observation Equation
        """
        obs_matrix = [
            states[0] + states[2],
            - self.gravity * np.sin(states[1]),
        ]
        return np.asarray(obs_matrix)

    def jacobian_obs_equation(self, states):
        """
        Jacobian matrix of observation Equation
        """
        jac = [
            [1, 0, 1],
            [0, - self.gravity * np.cos(states[1]), 0]
        ]
        return np.asarray(jac)
