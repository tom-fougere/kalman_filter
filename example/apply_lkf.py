import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from linear.linear_kalman_filter import LinearKalmanFilter

PLOT_FIG = False


def example_1():
    """
    Demonstrate how to use the Linear Kalman filter
    :return:
    """

    # Parameters of sinus wave
    f = 20  # Hz
    amplitude = 3
    noise_std_level = 0.5
    sampling_frequency_hz = 1000
    dt = 1/sampling_frequency_hz

    # Sinus wave
    time = np.linspace(0, 1, sampling_frequency_hz).reshape(1, -1)
    signal = amplitude * np.sin(2 * math.pi * f * time) + noise_std_level * np.random.randn(1, time.shape[1])

    # Setup LKF
    lkf = LinearKalmanFilter(1, 1, time.shape[1])

    # Set parameters
    lkf.set_state_equation(np.asarray([[1]]), np.asarray([[noise_std_level**2]]))
    lkf.set_obs_equation(np.asarray([[1]]), np.asarray([[1]]))

    # Set the initial states of the signal
    lkf.init(signal[:, 0], 1)

    # Run filter
    filtered_states = lkf.filter(signal)

    # Plot Result
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time[0, :],
            y=signal[0, :],
            name='Signal to filter',
        ))
    fig.add_trace(
        go.Scatter(
            x=time[0, :],
            y=filtered_states[0, :],
            name='Filtered signal',
        )
    )
    fig.update_layout(
        title='Example 1',
        xaxis_title='Time',
    )

    if PLOT_FIG:
        fig.show()


def example_2():
    ## Second example
    # from: http://www.ferdinandpiette.com/blog/2011/04/exemple-dutilisation-du-filtre-de-kalman/

    fe = 50  # sampling frequency
    time_duration = 10  # Duration of measurement
    time = np.linspace(0, fe, fe*time_duration)

    # Computation of states (it's normally the system unknown !) [va ; a; b]
    real_states = np.zeros((time_duration*fe, 3))
    real_states[:, 0] = math.pi * 2 * math.pi * np.sin(2 * math.pi * time.T)
    real_states[:, 1] = -math.pi * np.cos(2 * math.pi * time.T) + math.pi
    real_states[:, 2] = 10 * time.T + 20 * np.sin(2 * math.pi * 0.1 * time.T)

    # Noise of the sensors (standard deviation)
    sensor_noise = np.zeros(2)
    sensor_noise[0] = 2 * math.pi**2 * 0.03
    sensor_noise[1] = 2 * math.pi * 0.1

    # Generation of measurements [va; a]
    measurements = np.zeros((time_duration * fe, 2))
    measurements[:, 0] = real_states[:, 0] + real_states[:, 2] + sensor_noise[0] * np.random.randn(time_duration * fe,)
    measurements[:, 1] = real_states[:, 1] + sensor_noise[1] * np.random.randn(time_duration*fe,)

    # Setup LKF
    lkf = LinearKalmanFilter(
        dim_state_vector=3,
        dim_obs_vector=2,
        dim_time=time.shape[0])

    # Set parameters
    state_matrix = [
        [1, 0, 0],
        [1/fe, 1, 0],
        [0, 0, 1]]
    cov_state = [
        [100, 0, 0],
        [0, 0, 0],
        [0, 0, 2]]
    obs_matrix = [
        [1, 0, 1],
        [0, 1, 0],
    ]
    cov_obs = [
        [sensor_noise[0]**2, 0],
        [0, sensor_noise[1]**2],
    ]
    lkf.set_state_equation(np.asarray(state_matrix), np.asarray(cov_state))
    lkf.set_obs_equation(np.asarray(obs_matrix), np.asarray(cov_obs))

    # Set the initial states of the signal
    lkf.init(np.asarray([0, 0, 0]), np.zeros((3, 3)))

    # Run filter
    filtered_states = lkf.filter(measurements.T)

    # Plot Result
    fig = make_subplots(rows=3, cols=1)
    # State 1
    fig.add_trace(
        go.Scatter(
            x=time,
            y=real_states[:, 0],
            name='Reality',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=measurements[:, 0],
            name='Measurements',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=filtered_states[0, :],
            name='Estimation (LKF)',
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        title='State 1',
        xaxis_title='Time',
    )

    # State 2
    fig.add_trace(
        go.Scatter(
            x=time,
            y=real_states[:, 1],
            name='Reality',
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=measurements[:, 1],
            name='Measurements',
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=filtered_states[1, :],
            name='Estimation (LKF)',
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title='State 2',
        xaxis_title='Time',
    )

    # State 3
    fig.add_trace(
        go.Scatter(
            x=time,
            y=real_states[:, 2],
            name='Reality',
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=filtered_states[2, :],
            name='Estimation (LKF)',
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        title='State 3',
        xaxis_title='Time',
    )

    if PLOT_FIG:
        fig.show()


if __name__ == '__main__':
    example_1()
    example_2()
