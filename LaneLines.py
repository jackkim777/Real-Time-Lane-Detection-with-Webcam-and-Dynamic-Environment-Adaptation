import cv2
import numpy as np
import matplotlib.image as mpimg


kp = 0.1
ki = 0.0001
kd = 0.1


def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)
class LaneLines:
    """ Class containing information about detected lane lines.
    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        debug (boolean): Flag for debug/normal mode
    """
    def __init__(self):
        """Init Lanelines.
        Parameters:
            left_fit (np.array): Coefficients of polynomial that fit left lane
            right_fit (np.array): Coefficients of polynomial that fit right lane
            binary (np.array): binary image
        """
        self.left_fit = None
        self.right_fit = None
        self.prev_left_fit = None  # 저장: 이전 프레임의 왼쪽 차선 폴리노미얼 계수
        self.prev_right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.keep_straight_img = mpimg.imread('straight.png')
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = 9
        # Width of the the windows +/- margin
        self.margin = 100
        # Mininum number of pixels found to recenter window
        self.minpix = 50

        tl = (50, 415)  
        bl = (0 , 450)
        tr = (590, 415)
        br = (640, 450)
        self.src_points = np.float32([tl, bl, tr, br])
        self.dst_points = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
        self.P = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def forward(self, img):
        """Take a image and detect lane lines.
        Parameters:
            img (np.array): An binary image containing relevant pixels
        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        if self.left_fit is None:
            # 예를 들어, 이미지 너비의 1/4 지점을 왼쪽 차선의 시작점으로 가정
            self.left_fit = np.array([0, 0, img.shape[1] / 8])
        if self.right_fit is None:
            # 예를 들어, 이미지 너비의 3/4 지점을 오른쪽 차선의 시작점으로 가정
            self.right_fit = np.array([0, 0, 7 * img.shape[1] / 8])
        self.extract_features(img)
        # PID
        frame = self.fit_poly(img)

        cmdvelValue = 0 

        global kp, ki, kd

        # bird_eye_view = cv2.warpPerspective(frame, self.P, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([120, 255, 255])
        
        # # 흰색 차선을 위한 HSV 범위
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # # 노란색 차선을 위한 HSV 범위
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # # 두 마스크를 결합하여 새로운 마스크 생성
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        res = cv2.bitwise_and(frame_rgb, frame_rgb, mask=combined_mask)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


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
                # print(cmdvelValue)
                # self.cmdvelPub(cmdvelValue)
                
                cv2.line(frame_rgb, (cx, cy), (frame_rgb.shape[1] // 2, cy), (0, 0, 255), 5)
                # cv2.imshow('PID Frame', frame_rgb)
        # return self.fit_poly(img)
        return frame_rgb, cmdvelValue

    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that in a specific window
        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window
        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)
        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]
    def extract_features(self, img):
        """ Extract features from a binary image
        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        # Height of of windows - based on nwindows and image shape
        self.window_height = int(img.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixel in the image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])
    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image.
        Parameters:
            img (np.array): A binary warped image
        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            out_img (np.array): A RGB image that use to display result later on.
        """
        assert(len(img.shape) == 2)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))
        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Current position to be update later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2
        # Create empty lists to reveice left and right lane pixel
        leftx, lefty, rightx, righty = [], [], [], []
        # Step through the windows one by one
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)
            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)
            # Append these indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)
            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))
        return leftx, lefty, rightx, righty, out_img
    def fit_poly(self, img):
        """Find the lane line from an image and draw it.
        Parameters:
            img (np.array): a binary warped image
        Returns:
            out_img (np.array): a RGB image that have lane line drawn on that.
        """
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)
        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)
        if self.left_fit is None or self.right_fit is None:
        # left_fit 또는 right_fit이 None이면 계산을 건너뛰고 원본 이미지 반환
            return out_img
        # Generatex and  y values for plotting
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))
        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))
        ploty = np.linspace(miny, maxy, img.shape[0])
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        # Visualization
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))
            center_x = (l + r) // 2
            cv2.line(out_img, (center_x, y), (center_x, y), (255, 0, 0), thickness=5)
        self.center_x = center_x
        self.y = y
        
        # print(f"Left: {int(left_fitx[0])}, Right: {int(right_fitx[0])}")
        lR, rR, pos = self.measure_curvature()
        return out_img
    def plot(self, out_img):
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature()
        if self.left_fit is None or self.right_fit is None:
            # 예외 처리: left_fit 또는 right_fit이 None인 경우 경고 메시지 출력하고 원본 이미지 반환
            cv2.putText(out_img, "Lane detection unavailable", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return out_img
        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]
        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')
        if len(self.dir) > 10:
            self.dir.pop(0)
        W = 400
        H = 500
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0,:] = [0, 0, 255]
        widget[-1,:] = [0, 0, 255]
        widget[:,0] = [0, 0, 255]
        widget[:,-1] = [0, 0, 255]
        out_img[:H, :W] = widget
        direction = max(set(self.dir), key = self.dir.count)
        msg = "Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
        if direction == 'L':
            y, x = self.left_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"
        if direction == 'R':
            y, x = self.right_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"
        if direction == 'F':
            y, x = self.keep_straight_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.keep_straight_img[y, x, :3]
        cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.putText(
            out_img,
            "Good Lane Keeping",
            org=(10, 400),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(0, 255, 0),
            thickness=2)
        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m away from center".format(pos),
            org=(10, 450),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.66,
            color=(255, 255, 255),
            thickness=2)
        return out_img
    def measure_curvature(self):
        ym = 30/720
        xm = 3.7/700
        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym
        # Compute R_curve (radius of curvature)
        left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
        right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        xl = np.dot(self.left_fit, [700**2, 700, 1])
        xr = np.dot(self.right_fit, [700**2, 700, 1])
        pos = (1280//2 - (xl+xr)//2)*xm
        return left_curveR, right_curveR, pos