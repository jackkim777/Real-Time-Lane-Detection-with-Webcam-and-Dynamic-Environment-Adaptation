import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class CameraCalibration():
    def __init__(self, image_dir, nx, ny, debug=False):
        fnames = glob.glob("{}/*".format(image_dir))
        objpoints = []
        imgpoints = []
        
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        for f in fnames:
            img = cv2.imread(f)
            if img is None:
                continue
            # 올바르게 BGR에서 GRAY로 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (nx, ny))  # gray 이미지 사용
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)

        if len(objpoints) == 0:
            raise Exception("No chessboard corners found in any image files.")

        shape = gray.shape[::-1]  # gray 이미지의 shape 사용
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

        if not ret:
            raise Exception("Unable to calibrate camera")

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
# class CameraCalibration():
#     """ Class that calibrate camera using chessboard images.

#     Attributes:
#         mtx (np.array): Camera matrix 
#         dist (np.array): Distortion coefficients
#     """
    # def __init__(self, image_dir, nx, ny, debug=False):
    #     """ Init CameraCalibration.

    #     Parameters:
    #         image_dir (str): path to folder contains chessboard images
    #         nx (int): width of chessboard (number of squares)
    #         ny (int): height of chessboard (number of squares)
    #     """
    #     fnames = glob.glob("{}/*".format(image_dir))
    #     objpoints = []
    #     imgpoints = []
        
    #     # Coordinates of chessboard's corners in 3D
    #     objp = np.zeros((nx*ny, 3), np.float32)
    #     objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
    #     # Go through all chessboard images
    #     for f in fnames:
    #         img = mpimg.imread(f)

    #         # Convert to grayscale image
    #         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #         # Find chessboard corners
    #         ret, corners = cv2.findChessboardCorners(img, (nx, ny))
    #         if ret:
    #             imgpoints.append(corners)
    #             objpoints.append(objp)

    #     shape = (img.shape[1], img.shape[0])
    #     ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

    #     if not ret:
    #         raise Exception("Unable to calibrate camera")

    def undistort(self, img):
        """ Return undistort image.

        Parameters:
            img (np.array): Input image

        Returns:
            Image (np.array): Undistorted image
        """
        # Convert to grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
