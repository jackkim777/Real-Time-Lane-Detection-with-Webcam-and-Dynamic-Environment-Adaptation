import numpy as np
import cv2
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from geometry_msgs.msg import Twist
import rospy
import signal
import sys

kp = 0.1
ki = 0.0001
kd = 0.1

class FindLaneLines:
    def __init__(self):
        """ Init Application"""
        rospy.init_node('cmd_vel_publisher', anonymous=True)
        self.cmdvelPublisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.Twist_msg = Twist()

        self.calibration = CameraCalibration('/home/erp-42/test', 8, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

        tl = (50, 415)
        bl = (0 , 450)
        tr = (590, 415)
        br = (640, 450)
        self.src_points = np.float32([tl, bl, tr, br])
        self.dst_points = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
        self.P = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def cmdvelPub(self, cmdvelValue):
        self.Twist_msg.angular.z = cmdvelValue
        self.Twist_msg.linear.x = 0.3
        self.cmdvelPublisher.publish(self.Twist_msg)
        rospy.loginfo("Publishing velocity: linear=%f m/s, angular=%f rad/s", self.Twist_msg.linear.x, self.Twist_msg.angular.z)
    
    def forward(self, img):
        # 원본 이미지를 복사하여 결과 이미지 준비
        resized_img = cv2.resize(img, (1280, 720))
        out_img = np.copy(resized_img)

        # 카메라 교정 적용
        undistorted_img = self.calibration.undistort(resized_img)
        # cv2.imshow('Undistorted Image1', undistorted_img)
        # # 투시 변환 적용
        transformed_img = self.transform.forward(undistorted_img)
        cv2.imshow('Perspective Transformed Image2', transformed_img)

        # 임계값 적용
        thresholded_img = self.thresholding.forward(transformed_img)
        cv2.imshow('Thresholded Image3', thresholded_img)

        # 차선 인식
        lane_img, cmdvelValue = self.lanelines.forward(thresholded_img)
        cv2.imshow('Lane Image4', lane_img)

        # 투시 변환 복원
        back_transformed_img = self.transform.backward(lane_img)

        # 크기 조정
        if back_transformed_img.shape != out_img.shape:
            back_transformed_img = cv2.resize(back_transformed_img, (out_img.shape[1], out_img.shape[0]))

        # 원본 이미지에 라인 그리기
        out_img = cv2.addWeighted(out_img, 1, back_transformed_img, 0.6, 0)
        final_img = self.lanelines.plot(out_img)
        # cv2.imshow('Final Output Image5', final_img)

        return final_img, cmdvelValue

    def PID_controll(self,  frame):
        global kp, ki, kd

        bird_eye_view = cv2.warpPerspective(frame, self.P, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(bird_eye_view, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([120, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            if M['m00'] != 0: # PID 제어 부분
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                error = cx - frame_rgb.shape[1] // 2
                steering_angle = kp * error
                cmdvelValue = -steering_angle * 0.035
                self.cmdvelValue = cmdvelValue
                print(cmdvelValue)
                # self.cmdvelPub(cmdvelValue)
                
                cv2.line(frame_rgb, (cx, cy), (frame_rgb.shape[1] // 2, cy), (0, 0, 255), 5)
                cv2.imshow('PID Frame', frame_rgb)

def signal_handler(signal, frame):
    print('SIGINT caught, shutting down ROS node')
    rospy.signal_shutdown('SIGINT caught')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    findLaneLines = FindLaneLines()
    frame = cv2.imread('test_img.jpg')

    if frame is None:
        print("이미지를 불러올 수 없습니다.")
        return

    # 차선 탐지 로직 적용
    processed_frame, cmdVelValue = findLaneLines.forward(frame)
    # 처리된 이미지 표시
    print(f"CmdVel Value : {cmdVelValue}")
    cv2.imshow('Processed Frame', processed_frame)
    findLaneLines.cmdvelPub(cmdVelValue)

    # 'q' 키를 눌러 종료
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 시 자원 해제
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
