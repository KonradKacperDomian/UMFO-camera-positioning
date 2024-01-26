__title__       = "IMU"                                                               
__author__      = "Konrad Kacper Domian"
__copyright__   = "Participants of UMFO"              
__license__     = "For internal use only"                                                   
__version__     = "060124"                  # 6 January 2024                                                       
__maintainer__  = "Konrad Kacper Domian"                                                          
__status__      = "Production"

'''
Class for using IMU
'''

# imports
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ACCELEROMETER, BNO_REPORT_ROTATION_VECTOR
import threading
import math
import time

class IMU_thread(threading.Thread):

    def __init__(self, terminal_print=True):
        self.terminal_print = terminal_print
        super(IMU_thread, self).__init__()                  # Create new thread
        self.i2c = busio.I2C(3, 2)                          # I2C connect to 3 pin and 2 pin
        self.bno = BNO08X_I2C(self.i2c)                     # Initialize BNO08X driver
        self.bno.initialize()
        self.accel_sampling_period = 1/500                  # Maximum sampling rate of Accelerometer

        self.current_speed = [0.0, 0.0, 0.0]
        self.current_position = [0.0, 0.0, 0.0]
        self.current_rotation = [0.0, 0.0, 0.0]
        self.current_movement = [0.0, 0.0, 0.0]

    def print_data(self, movement, rotation, position):
        roll_deg, pitch_deg, yaw_deg = rotation
        X, Y, Z = movement
        x, y, z = position
        print("")
        print("Rotation: Roll: %0.6f  Pitch: %0.6f Yaw: %0.6f  degree" % (roll_deg, pitch_deg, yaw_deg))
        print("Movement: X: %0.6f  Y: %0.6f Z: %0.6f  m" % (X, Y, Z))
        print("Position: x: %0.6f  y: %0.6f z: %0.6f  m" % (x, y, z))

    def convert_quaternion_to_euler_degrees(self, quaternion):
        quat_i, quat_j, quat_k, quat_real = quaternion
        try:
            # Roll (φ)
            roll = math.atan2(2 * (quat_i * quat_real + quat_j * quat_k), 1 - 2 * (quat_i**2 + quat_j**2))
            # Pitch (θ)
            pitch = math.asin(2 * (quat_i * quat_k - quat_real * quat_j))
            # Yaw (ψ)
            yaw = math.atan2(2 * (quat_i * quat_j + quat_k * quat_real), 1 - 2 * (quat_j**2 + quat_k**2))
        except:
            print("Warming! Data not valid!")
            roll = 0.0
            pitch = 0.0
            yaw = 0.0
        # Convert radians to degrees if needed
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        return roll_deg, pitch_deg, yaw_deg

    def calc_current_speed(self, current_speed, accelerometer):
        accel_x, accel_y, accel_z = accelerometer
        speed_x, speed_y, speed_z = current_speed
        # Speed X
        speed_x = accel_x * self.accel_sampling_period
        # Speed Y
        speed_y = accel_y * self.accel_sampling_period
        # Speed Z
        speed_z = accel_z * self.accel_sampling_period

        return speed_x, speed_y, speed_z

    def convert_accelerometer_to_movement(self, accelerometer, current_speed):
        accel_x, accel_y, accel_z = accelerometer
        speed_x, speed_y, speed_z = current_speed
        # Movement X
        X = speed_x * self.accel_sampling_period + 0.5 * accel_x * self.accel_sampling_period ** 2 
        # Movement Y
        Y = speed_y * self.accel_sampling_period + 0.5 * accel_y * self.accel_sampling_period ** 2 
        # Movement Z
        Z = speed_z * self.accel_sampling_period + 0.5 * accel_z * self.accel_sampling_period ** 2 

        return X, Y, Z

    def change_position(self, Movement):
        X, Y, Z = Movement
        x, y, z = self.current_position
        x += X
        y += Y
        z += Z
        return x, y, z

    def calibration(self):
        self.bno.begin_calibration()
        self.bno.save_calibration_data() 

    def run(self):
        self.bno.enable_feature(BNO_REPORT_ACCELEROMETER)   # Enable Acceleroment feature 
        self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR) # Enable Rotation Quaternion
        while True:
            accelerometer = self.bno.acceleration
            quaternion = self.bno.quaternion
            self.rotation = self.convert_quaternion_to_euler_degrees(quaternion)

            self.current_movement = self.convert_accelerometer_to_movement(accelerometer, self.current_speed)
            self.current_speed = self.calc_current_speed(self.current_speed, accelerometer)
            self.current_position = self.change_position(self.current_movement)
            if self.terminal_print:
                self.print_data(self.current_movement, self.rotation, self.current_position)

# Example of Use!
if __name__ == "__main__":
    imu = IMU_thread(terminal_print = False)
    # IMU Calibration
    print("Start Calibration")
    imu.calibration()
    print("Finish Calibration")
    imu.start()

    while True:
        time.sleep(1)
        imu.print_data(imu.current_movement, imu.rotation, imu.current_position)