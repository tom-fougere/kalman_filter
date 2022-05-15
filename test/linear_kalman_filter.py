import unittest
from linear.linear_kalman_filter import *


class MyTestCase(unittest.TestCase):
    def test_constant_signal_wo_noise(self):
        # Parameters of sinus wave
        offset = 3
        sampling_frequency_hz = 1000

        # Constant signal
        time = np.linspace(0, 1, sampling_frequency_hz)
        signal = offset * np.ones((1, len(time)))

        # Setup LKF
        lkf = LinearKalmanFilter(1, 1, len(time))

        # Set parameters
        lkf.set_state_equation(np.asarray([[1]]), np.asarray([[0]]))
        lkf.set_obs_equation(np.asarray([[1]]), np.asarray([[1]]))

        # Set the initial states of the signal
        lkf.init(offset, np.asarray([[0]]))

        # Run filter
        filtered_states = lkf.filter(signal)

        np.testing.assert_array_equal(filtered_states, signal)

    def test_constant_noisy_signal(self):
        # Parameters
        offset = 12
        sampling_frequency_hz = 1000
        noise_std = 3

        # Constant signal
        time = np.linspace(0, 1, sampling_frequency_hz)
        signal = offset * np.ones((1, len(time))) + noise_std * np.random.randn(len(time))

        # Setup LKF
        lkf = LinearKalmanFilter(1, 1, len(time))

        # Set parameters
        lkf.set_state_equation(np.asarray([[1]]), np.asarray([[0]]))
        lkf.set_obs_equation(np.asarray([[1]]), np.asarray([[noise_std]]))

        # Set the initial states of the signal
        init_value = offset + 5
        lkf.init(init_value, 1)

        # Run filter
        filtered_states = lkf.filter(signal)

        tolerance = 1
        self.assertGreater(np.std(signal), np.std(filtered_states))
        self.assertGreater(filtered_states[0, -1], offset - tolerance)
        self.assertGreater(offset + tolerance, filtered_states[0, -1])
        self.assertGreater(filtered_states[0, 0], init_value - tolerance)
        self.assertGreater(init_value + tolerance, filtered_states[0, 0])


if __name__ == '__main__':
    unittest.main()
