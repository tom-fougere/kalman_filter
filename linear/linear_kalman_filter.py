import numpy as np


class LinearKalmanFilter:

    # Dimension of state vector
    dim_state_vector = 0
    # Dimension of observation vector
    dim_obs_vector = 0
    # Duration of the measurements
    dim_time = 0

    # Estimated states
    states = None
    # Innovation
    innovation = None
    # K: Optimal Kalmaan gain
    K = None

    # P: Covariance of states
    P = None
    # F: State-transition model
    F = None
    # B: Control-input model
    B = None
    # Q: Covariance of state (process) model
    Q = None
    # H: Observation model
    H = None
    # R: Covariance of observation noise
    R = None

    def __init__(self, dim_state_vector, dim_obs_vector, dim_time):
        """
        Initialization of the Linear Kalman Filter
        :param dim_state_vector: (float) Number of state, 0 < dim_state_vector
        :param dim_obs_vector: (float) Number of observations,  0 < dimObsVector
        :param dim_time: Number of samples in the signal to filter (duration), 0 < dimTime
        """

        # Set dimensions
        self.dim_state_vector = dim_state_vector
        self.dim_obs_vector = dim_obs_vector
        self.dim_time = dim_time

        # Initialization of matrices
        self.states = np.zeros((dim_state_vector, dim_time))
        self.innovation = np.zeros((dim_obs_vector, dim_time))
        self.K = np.zeros((dim_state_vector, dim_obs_vector, dim_time))
        self.P = np.zeros((dim_state_vector, dim_state_vector, dim_time))
        self.F = np.zeros((dim_state_vector, dim_state_vector))
        self.B = np.zeros((dim_state_vector, dim_state_vector))
        self.H = np.zeros((dim_obs_vector, dim_state_vector))
        self.Q = np.zeros((dim_state_vector, dim_state_vector))
        self.R = np.zeros((dim_obs_vector, dim_obs_vector))

    def init(self, init_states, init_cov):
        """
        Initialize the Kalman filter
        Set the initial estimations of states and associated covariance
        of states and the associated covariance
        :param init_states: (Array [1, dimStateVector]) Estimation of the initial states
        :param init_cov: (Array [dimStateVector, dimStateVector]) Estimation of the covariance of these initial states
        """

        # Beginning of the first step
        self.P[:, :, 0] = init_cov
        self.states[:, 0] = init_states
        inverse_matrix = np.linalg.inv(self.H.dot(self.P[:, :, 0]).dot(self.H.T) + self.R)
        self.K[:, :, 0] = self.P[:, :, 0].dot(self.H.T).dot(inverse_matrix)

    def set_state_equation(self, f_mat, q_mat, b_mat=None):
        """
        Set the state equation
        state(k+1) = F * state(k) + B * command(k) + w(k)
        w is the process noise with covariance Q

        @todo: be careful, b_mat is not well managed, let it empty
        :param f_mat: (Array [dimStateVector, dimStateVector]) State transition model (state model or process model)
        :param q_mat: (Array [dimStateVector, dimStateVector]) Covariance of the process noise
        :param b_mat: (Array [dimStateVector, X]) (Optional) Control-input model, default to 0
        """

        # Set matrices
        self.F = f_mat
        self.Q = q_mat

        if b_mat:
            self.B = b_mat

    def set_obs_equation(self, h_mat, r_mat):
        """
        Set the observation equation
        obs(k) = H * state(k) + v(k)
        v is the observation noise with covariance R
        :param h_mat: (Array [dimObsVector, dimStateVector]) Observation model (state model or process model)
        :param r_mat: (Array [dimObsVector, dimObsVector]) Covariance of observation noise
        """

        # Set matrices
        self.H = h_mat
        self.R = r_mat

    def filter(self, measurements, commands=None):
        """
        Run the Linear Kalman Filter on the given measurements
        :param measurements: (Array [dimObsVector, dimTime]) Measurements to filter
        :param commands: (Array [dimObsVector, dimTime]) (Optional) Known control inputs of the system (or commands), default to 0
        :return: (Array [dimStateVector, dimTime]) Estimations of the states
        """

        # Command creation
        if not commands:
            commands = 0.

        # Continuing the first step
        self.innovation[:, 0] = measurements[:, 0] - self.H.dot(self.states[:, 0])

        # Iteration Kalman operations
        self._iterate(measurements, commands)

        # Broadcast outputs
        states = self.states

        return states

    def _iterate(self, measurements, commands):
        """
        Iterate the Kalman Filter operations (prediction / correction)
        :param measurements: (Array [dimObsVector, dimTime]) Observed measurements
        :param commands: (Array [dimObsVector, dimTime]) Commands
        """

        for k in range(self.dim_time-1):

            # Prediction
            self.states[:, k+1] = self.F.dot(self.states[:, k])  # + self.B.dot(commands[:, k])
            self.P[:, :, k+1] = self.F.dot(self.P[:, :, k]).dot(self.F.T) + self.Q

            # Correction
            inverse_matrix = np.linalg.inv(self.H.dot(self.P[:, :, k+1]).dot(self.H.T) + self.R)
            self.K[:, :, k+1] = self.P[:, :, k+1].dot(self.H.T).dot(inverse_matrix)
            self.innovation[:, k+1] = measurements[:, k+1] - self.H.dot(self.states[:, k+1])
            self.states[:, k+1] = self.states[:, k+1] + self.K[:, :, k+1].dot(self.innovation[:, k+1])
            self.P[:, :, k+1] = (np.eye(self.dim_state_vector) - self.K[:, :, k+1].dot(self.H)).dot(self.P[:, :, k+1])
