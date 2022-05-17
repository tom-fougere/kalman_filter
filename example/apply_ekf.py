import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from extended.extended_kalman_filter import ExtendedKalmanFilter
from example.equations_example import ExampleEquation

PLOT_FIG = False


def example():
    # Example
    # from: http://www.ferdinandpiette.com/blog/2011/04/exemple-dutilisation-du-filtre-de-kalman/

    gravity = 1  # m/s²
    fe = 10  # sampling frequency
    time_duration = 100  # Duration of measurement
    time = np.linspace(0, fe, fe*time_duration)

    # Computation of states (it's normally the system unknown !) [va; a; b]
    real_states = np.zeros((time_duration*fe, 3))
    real_states[:, 0] = 2 * math.pi * np.sin(2 * math.pi * time.T)
    real_states[:, 1] = -np.cos(2 * math.pi * time.T) + math.pi
    real_states[:, 2] = 10 * time.T + 20 * np.sin(2 * math.pi * 0.1 * time.T)

    # Noise of the sensors (standard deviation)
    sensor_noise = np.zeros(2)
    sensor_noise[0] = 0.5
    sensor_noise[1] = 0.1

    # Generation of measurements [va; a]
    measurements = np.zeros((time_duration * fe, 2))
    measurements[:, 0] = real_states[:, 0] + real_states[:, 2] + sensor_noise[0] * np.random.randn(time_duration * fe,)
    measurements[:, 1] = -gravity * np.sin(real_states[:, 1]) + sensor_noise[1] * np.random.randn(time_duration * fe,)

    # Setup LKF
    lkf = ExtendedKalmanFilter(
        dim_state_vector=3,
        dim_obs_vector=2,
        dim_time=time.shape[0])

    # Set parameters
    kalman_equations = ExampleEquation(gravity=gravity, Te=1/fe)
    cov_state = [
        [100, 0, 0],
        [0, 0, 0],
        [0, 0, 2]]
    cov_obs = [
        [sensor_noise[0]**2, 0],
        [0, sensor_noise[1]**2],
    ]
    lkf.set_equations(
        kalman_equations=kalman_equations,
        q_mat=np.asarray(cov_state),
        r_mat=np.asarray(cov_obs)
    )

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
            mode='markers',
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

    # if PLOT_FIG:
    fig.show()


if __name__ == '__main__':
    example()
