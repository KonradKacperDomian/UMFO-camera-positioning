import cv2
import numpy as np
import glob
import csv

def CameraCalib():
        # Define the dimensions of checkerboard
        CHECKERBOARD = (6,9)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Vector for 3D points
        threedpoints = []
        # Vector for 2D points
        twodpoints = []
        # 3D points real world coordinates
        objectp3d = np.zeros(
            (1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                       0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        # Extracting path of individual image stored
        # in a given directory. Since no path is
        # specified, it will take current directory
        # jpg files alone
        images = glob.glob('zdj/zdj*.jpg')
        for filename in images:
            image = cv2.imread(filename)
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            # If desired number of corners are
            # found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If desired number of corners can be detected then,
            # refine the pixel coordinates and display
            # them on the images of checker board
            if ret == True:
                threedpoints.append(objectp3d)

                # Refining pixel coordinates
                # for given 2d points.
                corners2 = cv2.cornerSubPix(
                    grayColor, corners, (11, 11), (-1, -1), criteria)

                twodpoints.append(corners2)

                # Draw and display the corners
                image = cv2.drawChessboardCorners(
                    image, CHECKERBOARD, corners2, ret)


        cv2.destroyAllWindows()

        # Perform camera calibration by
        # passing the value of above found out 3D points (threedpoints)
        # and its corresponding pixel coordinates of the
        # detected corners (twodpoints)
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None,
                                                                      None)
        # Displaying required output
        
        print(" Camera matrix:")
        print(matrix)
        # open the file in the write mode
        f = open('calibrationMatrix.csv', 'w')

        # create the csv writer
        writer = csv.writer(f)

        # write to the csv file
        writer.writerows(matrix)

        # close the file
        f.close()


CameraCalib()