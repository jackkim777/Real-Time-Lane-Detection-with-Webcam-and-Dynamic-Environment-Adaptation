import numpy as np
from CameraCalibration import CameraCalibration
import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge

# 교정할 이미지 로드
test_img = cv2.imread('/home/erp-42/catkin_ws/src/Advanced-Lane-Lines/test_images/Screenshot from 2024-04-04 13-30-22.png')

# 카메라 교정 객체 생성 및 이미지 교정
calib = CameraCalibration('/home/erp-42/test/', 8, 6)
undistorted_img = calib.undistort(test_img)

# 원본 이미지와 교정된 이미지 비교하여 표시
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Undistorted Image')
plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))

plt.show()
