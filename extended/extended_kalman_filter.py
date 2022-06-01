import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ExtendedKalmanFilter:

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
    # K: Optimal Kalman gain
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

    # Class containing state equation, observation equation and their jacobian matrices
    kalman_equations = None
    # # State Equation
    # f_state_equation = None
    # # Jacobian matrix of State Equation
    # f_jacobian_state_equation = None
    # # Observation Equation
    # f_obs_equation = None
    # # Jacobian matrix of observation equation
    # f_jacobian_obs_equation = None

    def __init__(self, dim_state_vector, dim_obs_vector, dim_time):
        """
        Initialization of the Extended Kalman Filter
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
        self.H = self.kalman_equations.jacobian_obs_equation(self.states[:, 0])
        inverse_matrix = np.linalg.inv(self.H.dot(self.P[:, :, 0]).dot(self.H.T) + self.R)
        self.K[:, :, 0] = self.P[:, :, 0].dot(self.H.T).dot(inverse_matrix)

    def set_equations(self, kalman_equations, q_mat, r_mat):
        """
        Set the state and observation equations

        State equation:
        state(k+1) = F(state(k)) + B * command(k) + w(k)
        w is the process noise with covariance Q

        Observation equation:
        obs(k) = H(state(k)) + v(k)
        v is the observation noise with covariance R

        :param kalman_equations: (Class KalmanEquation) Class containing state and observation equations
                                                        and their jacobian
        :param q_mat: (Array [dimStateVector, dimStateVector]) Covariance of the process noise
        :param r_mat: (Array [dimObsVector, dimObsVector]) Covariance of observation noise
        """

        # Set matrices
        self.kalman_equations = kalman_equations
        self.Q = q_mat
        self.R = r_mat

    def filter(self, measurements):

        """
        Run the Extended Kalman Filter on the given measurements
        :param measurements: (Array [dimObsVector, dimTime]) Measurements to filter
        :return: (Array [dimStateVector, dimTime]) Estimations of the states
        """

        # Continuing the first step
        self.innovation[:, 0] = measurements[:, 0] - self.H.dot(self.states[:, 0])

        # Iteration Kalman operations
        self._iterate(measurements)

        # Broadcast outputs
        states = self.states

        return states

    def _iterate(self, measurements):
        """
        Iterate the Extended Kalman Filter operations (prediction / correction)
        :param measurements: (Array [dimObsVector, dimTime]) Observed measurements
        """

        for k in range(self.dim_time-1):

            # Prediction
            self.F = self.kalman_equations.jacobian_state_equation(self.states[:, k])
            self.states[:, k+1] = self.kalman_equations.state_equation(self.states[:, k])
            self.P[:, :, k+1] = self.F.dot(self.P[:, :, k]).dot(self.F.T) + self.Q

            # Correction
            self.H = self.kalman_equations.jacobian_obs_equation(self.states[:, k])
            inverse_matrix = np.linalg.inv(self.H.dot(self.P[:, :, k+1]).dot(self.H.T) + self.R)
            self.K[:, :, k+1] = self.P[:, :, k+1].dot(self.H.T).dot(inverse_matrix)
            self.innovation[:, k+1] = measurements[:, k+1] - self.kalman_equations.obs_equation(self.states[:, k+1])
            self.states[:, k+1] = self.states[:, k+1] + self.K[:, :, k+1].dot(self.innovation[:, k+1])
            self.P[:, :, k+1] = (np.eye(self.dim_state_vector) - self.K[:, :, k+1].dot(self.H)).dot(self.P[:, :, k+1])

    def plot_inner_matrices(self):
        """
        Plot inner matrices of the current kalman filter
        The graph will contain the values of P, K and Innovation
        :return:
        """
        figure = make_subplots(rows=3, cols=1,
                               shared_xaxes=True,
                               subplot_titles=("P matrix", "K matrix", "Innovation"))

        x = np.linspace(0, self.dim_time, num=self.dim_time, endpoint=False)

        # P matrix
        for row in range(self.dim_state_vector):
            for column in range(self.dim_state_vector):
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=self.P[row, column, :],
                        name='P(' + str(row) + ', ' + str(column) + ')',
                    ),
                    row=1,
                    col=1,
                )

        # K matrix
        for row in range(self.dim_state_vector):
            for column in range(self.dim_obs_vector):
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=self.K[row, column, :],
                        name='K(' + str(row) + ', ' + str(column) + ')',
                    ),
                    row=2,
                    col=1,
                )

        # Innovation matrix
        for row in range(self.dim_obs_vector):
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=self.innovation[row, :],
                    name='Innovation(' + str(row) + ')',
                ),
                row=3,
                col=1,
            )

        figure.show()

