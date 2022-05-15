from abc import ABC, abstractmethod


class KalmanEquation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def state_equation(self):
        """
        State Equation
        """
        pass

    @abstractmethod
    def jacobian_state_equation(self):
        """
        Jacobian matrix of state Equation
        """
        pass

    @abstractmethod
    def obs_equation(self):
        """
        Observation Equation
        """
        pass

    @abstractmethod
    def jacobian_obs_equation(self):
        """
        Jacobian matrix of observation Equation
        """
        pass
