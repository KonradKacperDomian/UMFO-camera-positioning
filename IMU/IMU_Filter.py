
import numpy as np
import complementary_filter as cp


class IMU_Filter:

    def __init__(self, measurements_num=10, alpha=0.9, g_ref=(0., 0., 1.), theta_min=1e-6, highpass=.01, lowpass=.05):
        """
        Initialises the IMU filter class
        :param measurements_num: int, default 10
            Number of the past measurements that are fed to the filter
        :param alpha: float, default 0.9
            Weight of the angular velocity measurements in the estimate.
        :param g_ref: tuple, len 3, default (0., 0., 1.)
            Unit vector denoting direction of gravity.
        :param theta_min: float, default 1e-6
            Minimal angular velocity after filtering. Values smaller than this
            will be considered noise and are not used for the estimate.
        :param highpass: float, default .01
            Cutoff frequency of the high-pass filter for the angular velocity as
            fraction of Nyquist frequency.
        :param lowpass: float, default .05
            Cutoff frequency of the low-pass filter for the linear acceleration as
            fraction of Nyquist frequency.
        """
        self.accData = np.array(np.zeros((measurements_num, 3)))
        self.gyroData = np.array(np.zeros((measurements_num, 3)))
        self.timestamps = [sampling_period] * (measurements_num-1)
        self.alpha = alpha
        self.g_ref = g_ref
        self.theta_min = theta_min
        self.highpass = highpass
        self.lowpass = lowpass

    def get_next_value(self, timestamp, acceleration, rotation):
        """
        Filters incoming accelerometer and gyroscope data. Appends measurements to internal
        arrays that are then passed to a complementary filter.
        :param timestamp: Timestamp of current measurement (in s)
        :param acceleration: 3-element array with acceleration values (arbitrary values)
        :param rotation: 3-element array with angular velocity measurements (in rad/s).
        :return:
        q - Quaternion with the estimated orientation for each measurement.
        a - 3-axis acceleration filtered with lop-pass filter
        """
        # shift measurements by one time step
        self.accData = np.roll(self.accData, -1, axis=1)
        self.gyroData = np.roll(self.gyroData, -1, axis=1)

        # append new data
        self.accData[-1, 0] = acceleration[0]
        self.accData[-1, 1] = acceleration[1]
        self.accData[-1, 2] = acceleration[2]
        self.gyroData[-1, 0] = rotation[0]
        self.gyroData[-1, 1] = rotation[1]
        self.gyroData[-1, 2] = rotation[2]

        (q, a) = cp.estimate_orientation(a=self.accData,
                                         w=self.gyroData,
                                         t=self.timestamps,
                                         alpha=self.alpha,
                                         g_ref=self.g_ref,
                                         theta_min=self.theta_min,
                                         highpass=self.highpass,
                                         lowpass=self.lowpass)
        # return estimation of last measurements
        return q[-1], a[-1]

