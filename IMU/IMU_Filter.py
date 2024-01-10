
import numpy as np
import complementary_filter as cp


class IMU_Filter:

    def __init__(self, measurements_num, alpha, theta_min, highpass, lowpass):
        self.accData = np.array(np.zeros((measurements_num, 3)))
        self.gyroData = np.array(np.zeros((measurements_num, 3)))
        self.timestamps = np.array(np.zeros((measurements_num,)))
        self.alpha = alpha
        self.theta_min = theta_min
        self.highpass = highpass
        self.lowpass = lowpass

    def get_next_value(self, timestamp, acceleration, rotation):
        # shift measurements by one time step
        self.accData = np.roll(self.accData, -1, axis=1)
        self.gyroData = np.roll(self.gyroData, -1, axis=1)
        self.timestamps = np.roll(self.timestamps, -1)

        # append new data
        self.timestamps[-1] = timestamp
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
                                                 theta_min=self.theta_min,
                                                 highpass=self.highpass,
                                                 lowpass=self.lowpass)
        return q, a

